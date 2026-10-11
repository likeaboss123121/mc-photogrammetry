# DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-DESIGN-2026-10-04

Task ID: `DS-IMG-CANDIDATE-SUPPORT-PROVENANCE-DESIGN`. Owner: DeepSeek. Date:
2026-10-04. Revision: **R4 — ingress single-read chain closure: one encoded-byte
read/buffer shared by discovery and decode (2026-10-04 17:28 UTC Lead
clarification)**; supersedes R3, R2 and R1 (§9 retains the R1, R2 and R3
remediation maps and adds the R4 closure map). Status: **read-only source audit
complete; this file is the only artifact written. No source, test, capture,
output, solver call, truth, camera, mask, evaluator, image, Minecraft or JAR read
or write. No production change.**

Scope boundary: this trace uses source structure, dataclass fields and schema
text only. It never opens an image, mask, camera manifest, bound, reference
litematic, evaluator truth payload or JAR, and it makes no accuracy claim. It
answers one question: *which image-derived provenance currently reaches a plane
candidate (`PlaneMatch`), what the same-axis capture serializes, what the
evaluator-only renderer/ownership maps can join to, and what is lost.*

R1 revision note (retained). §1 and §7 (the read-only trace and its source
references) are retained. R1 made the three Luna HOLD conditions normative for
the implementation task and added the retrievable-payload, orientation, pin and
abstention requirements the Lead review asked for:

1. **Pre-filter coverage.** Every candidate admitted *before*
   `_apply_same_axis_filter` must be recorded with its keep/drop disposition and
   a bound witness or a typed witness refusal. Post-filter survivors alone
   cannot test a dropped alternative.
2. **Exact unique joins.** The 9-scalar `_input_match_identity` is not proven
   unique. Rows are joined by ordered candidate row index plus exact
   `match_sha256`/`envelope_sha256`, both endpoint signature digests and explicit
   alternative indices; `b` alone never resolves a join.
3. **Orientation and retrievable support.** `match_plane_pair` warps the left
   signature into the right grid (`plane_graph.py:804-810`) and logical pair
   order is left-source then right-target (`plane_graph.py:914-928`); source and
   target shapes must be recorded, asymmetric non-identity orientation tests are
   required, and the support payload must be retrievable and bounded — a digest
   alone cannot support pixel ownership attribution.

R2 revision note (retained). R1's Phase B joined
by **replaying `plane_signature` against the evaluator's rendered RGB** and
requiring signature equality (`evaluation_signature_sampling.py:122-179`). Luna
confirms that is **not safe for real Minecraft screenshots**: shader, lighting,
resource-pack and appearance differences change the signature arrays and the
correlation, so an evaluator re-render is not the screenshot the product
actually consumed. R2 therefore:

1. **Captures the sampler tap witness during the actual image-only signature
   call.** Phase A must run the ordinary signature sampler on the decoded
   screenshot and record the exact source-pixel tap/aggregation witness it
   produced (`SignatureSamplingWitnessV1`, `RectificationWitnessV1`), bound to
   the screenshot byte hash, decoded dimensions, decode/crop/resize transform,
   inference-side usable-domain digest, signature digest and sampler settings,
   **before** freeze. The witness is an output of the same call that mints the
   accepted signature — never re-derived, re-decoded or re-sampled later. §0
   makes this the primary contract.
2. **Projects evaluator ownership into that source-pixel space in Phase B.**
   Phase B maps the saved source-pixel indices to a separately pinned evaluator
   ownership map **projected through the evaluator camera into exactly the
   frozen source coordinate space**. It never replays the signature on evaluator
   RGB, never substitutes evaluator pixels for screenshot evidence, and never
   feeds an ownership result back into inference. §0 and §5.
3. **Refuses typed-unavailable when alignment cannot be proven** (dimensions,
   camera/intrinsics/pose, transform, occlusion), and keeps the R1 pre-filter
   coverage, exact unique joins, explicit left-source/right-target orientation,
   asymmetric non-identity tests and retrievable bounded support as mandatory
   conditions. §0, §3, §4, §6.

The Phase A image-only freeze then Phase B evaluator-only join, the
image/usable-domain pins, and the strict appearance-proxy abstention limits
remain mandatory. §9 records the R1 HOLD map, the R2 correction map, and the R3
closure map.

R3 revision note (newest Lead/Luna correction — normative). Luna's 2026-10-04
17:06 UTC R2 final-review HOLD found two implementation seams missing. R3 names
both in the design and makes them mandatory; it removes no R2 boundary.

1. **Pre-filter matcher-witness seam.** R2 already required every pre-filter
   admitted candidate to carry a witness or typed refusal (§3 M1), but it did not
   name *where* that witness is minted and published. The witness for every
   admitted match — including matches later dropped by `_apply_same_axis_filter`
   — is minted at matcher admission (`plane_graph.py:1036-1043`) and retained in
   the local `candidate_witnesses` table (`:1072-1076`), but the only publication
   point (`:1098-1132`) iterates `filtered` (survivors). R3 adds §0.5:
   publication must move to a full-universe pass over the sorted admitted rows,
   and each row must bind its immutable `admitted_row_index`, its keep/drop
   `disposition`, its post-filter `match_index` (or a typed absence), and its own
   witness/refusal. §3 M1, §3 tests (r)(s), §4 C13, §8 B, and §9.3 make this
   mandatory.
2. **Phase A ingress source pins.** R2 named the screenshot byte hash and
   `FrameTransform` as required Phase A pins, but described them as if they could
   be recovered at capture time. They cannot: `input.image.load_image`
   (`input/image.py:14-29`) opens the path, decodes, and discards the encoded
   bytes, and `build_image_derived_session` (`image_derived_session.py:280-316`)
   binds only the decoded-pixel digest. The byte SHA exists earlier as
   `SourceFrame.content_hash` (`input/media.py:60,84-89`) and is dropped by the
   entry (`image_derived_entry.py:448-455`); no `FrameTransform` is ever
   constructed on that path. R3 adds §0.6: a typed ingress record must capture
   the byte SHA-256/length at the single read that feeds the decode, co-locate it
   with the decoded-pixel digest and the reversible `FrameTransform`, and thread
   it through the session record into the signature call, with a typed refusal
   when ingress metadata is unavailable. §3 M4, §3 tests (t)(u), §4 C14, §8 A,
   and §9.3 make this mandatory. (R4 tightens "the single read that feeds the
   decode" to the full discovery/decode chain — see the R4 note below and §0.6
   item 1.)

All R2 boundaries are retained unchanged: the screenshot-sampler witness captured
in the ordinary signature call, the Phase B projection-only join that never
replays `plane_signature` on evaluator RGB and never feeds ownership back into
inference, the exact one-to-one joins, the output-neutral default path, the
bounded retrievable payloads, and the typed abstention/proxy limits.

R4 revision note (newest Lead clarification, normative). The 2026-10-04 17:28 UTC
Lead clarification found that R3's "read each screenshot once" was still not true
across the **full** ingress chain. R3 correctly said the byte pins must come from
the read that feeds the decode, but the source audit shows the ordinary chain
performs **two** separate physical reads of every input screenshot before any
signature exists:

1. **Discovery read.** `discover_media` (`input/media.py:42-81`) calls
   `_sha256(path)` (`:60`, `_sha256` `:84-89`) to read the entire encoded file and
   compute `SourceFrame.content_hash` (`:24`) for de-duplication
   (`seen_hashes`, `:56,61-63`); the bytes are then discarded.
2. **Decode read.** `input.image.load_image` (`input/image.py:14-29`) opens
   `source_path` **again** (`Image.open(source)`, `:25`) and decodes it; the entry
   loop `image = load_image(frame.source_path)` (`image_derived_entry.py:448-455`)
   keeps only the array and drops `content_hash`.

Capturing bytes only inside `load_image` would therefore still leave two physical
reads and would leave the discovery hash (`SourceFrame.content_hash`) computed
from a *different* read than the pixels. R4 closes that full-chain gap and makes
it mandatory that there is exactly **one** encoded-byte read/buffer per input
screenshot, shared by (a) the content hash/length and de-duplication and (b) the
decode, or, equivalently, that discovery performs no read at all and
`content_hash` is derived from the ingress record produced by that one read. No
later path may re-hash or re-open `source_path`, and the existing default
output/session behavior and typed-refusal boundary are preserved. §0.6 (item 1
and item 4), §0.2 item 1, §1.7, §2.13, §3 capture seam (i)/M4/tests (t)(u)(v),
§4 C14, §5, §6, §8 A/D and §9.4 make this mandatory.

## 0. Normative Phase A / Phase B contract (source-screenshot witness, ingress pins and pre-filter witness seam)

This section is the primary contract; §3–§6 instantiate it. It exists because the
product is fed an ordinary Minecraft screenshot, and the signature the matcher
scored was sampled from **that** screenshot's pixels.

### 0.1 The witness already exists inside the production sampler

`lattice_planes.plane_signature` (`lattice_planes.py:1664`) takes the caller's
`image` and `usable` and, **when and only when** a `witness_sink` is passed,
emits one `SignatureSamplingWitnessV1` (`lattice_planes.py:485`):

- `cell_source_ptr` / `cell_source_indices` (`:503-504`) — per-cell **source
  image flat pixel indices**; the `source_shape` carried by the nested
  `RectificationWitnessV1` (`lattice.py:585-646`, built at `lattice.py:655`) is
  `(height, width)` of the exact array that was sampled;
- nested `rgb_rectification` / `coverage_rectification` /
  `region_rectification` witnesses with `map_x/map_y/depth_valid/tap_ptr/
  tap_source_indices/tap_weight_numerators` — the exact OpenCV `INTER_LINEAR`
  fixed-point tap set;
- `grayscale_sha256`, `gradient_y_sha256`, `gradient_x_sha256`, `energy_sha256`,
  `coverage_patch_sha256`, `gradient_edge_order`, `subsample_factor`,
  `seam_band`, `cell_rectangles`, `interior_rectangles`, and
  `witness_sha256`.

The production call `derive_image_derived_signatures`
(`image_derived_signatures.py:410-413`) passes `image` and `usable` but **no**
`witness_sink`, so the ordinary record retains only the aggregate arrays and
`_signature_payload` (`:111`) does not include the witness. The witness is thus
already computable at the exact call that mints the accepted signature; it is
simply not captured. **Phase A must pass the witness sink at that call.**

### 0.2 Phase A — capture during the normal signature call, then freeze

Phase A runs the ordinary image-only pipeline. For every accepted
`SignatureEvidence`, the record must be produced by the **same** `plane_signature`
invocation that produced the accepted signature — same decoded array, same
`usable` mask, same sampler settings — and must bind, before freeze:

1. **Screenshot identity (ingress-bound; see §0.6).** The encoded file SHA-256
   and byte length captured at the single encoded-byte read/buffer that is shared
   by discovery (content hash, byte length and `SourceFrame` de-duplication) and
   by the decode, the decoded-RGB class/loader provenance
   (`input.image.load_image`, `input/image.py:14`, Pillow
   `ImageOps.exif_transpose(...).convert("RGB")`), the decoded array shape and
   `dtype`, and a digest over the exact decoded bytes
   (`image_derived_session.image_sha256`, `image_derived_session.py:264`).
   The byte hash/length must be bound to the *same* decoded array and session as
   the pixel digest and must be the hash of the very bytes the decoder consumed,
   never a discovery-side `_sha256(source_path)` from a separate read and never
   recomputed by reopening or re-hashing `source_path` after decode (§0.6).
2. **Dimensions and transform (ingress-bound; see §0.6).** The consumed image
   shape plus the reversible `FrameTransform` (`pipeline/contracts.py:13`) that
   maps screenshot source coordinates → consumed frame coordinates: `source_size`,
   `crop_origin`, `crop_size`, `output_size`/`resolved_output_size` and `scale`.
   The transform must be constructed from the same ingress decode that produced
   the bound byte hash and pixel digest. The current entry path decodes with no
   crop/resize (`image_derived_entry.py:448-458`), so the recorded transform is
   the identity; a non-identity transform must be recorded exactly, never assumed
   away, and a missing/uncaptured transform on the observer path is a typed
   refusal (§0.6), not a default identity.
3. **Inference-side usable domain.** The usable-mask record actually consumed
   (`UsableMask`, `image_derived_signatures.py:~90`): frame id, height/width,
   exact bool bytes and `mask_sha256`, plus the `masks_sha256` aggregate
   (`:195-197`). In the normal path this is
   `np.logical_not(create_gui_mask(h, w))` (`:389-393`); whatever produced it,
   the digest is pinned as inference-side truth about the sampled domain.
4. **Signature digest.** `SignatureEvidence.signature_sha256` over the canonical
   `_signature_payload` (`:111-120`).
5. **Sampler settings.** `max_cells`, `SIGNATURE_PIXELS_PER_CELL`,
   `MAX_RECTIFIED_SIDE`, the resolved `sub`/`pixels`/`cells_side`/`band`, the
   `WitnessBudgetV1` limits, and the rectification settings
   (`INTER_LINEAR`/`BORDER_CONSTANT_ZERO`, `weight_denominator=1024`,
   fixed-point tap encoding) — enough to show the witness was produced under the
   recorded settings.
6. **Source tap witness.** The `SignatureSamplingWitnessV1` itself, or a
   retrievable bounded handle to it, with `witness_sha256` recomputable from the
   record; the nested rectification witnesses carry `source_shape` equal to the
   consumed image shape.
7. **Binding checks.** `source_shape == consumed_image.shape[:2]`; every
   `cell_source_indices`/`tap_source_indices` entry is a valid flat index into
   that shape (or the `-1` border sentinel); the signature digest replays from
   the recorded settings and witness. A failure is a **typed refusal**
   (`source_tap_witness_unavailable`,
   `source_tap_witness_budget_exhausted`, `source_tap_witness_shape_mismatch`,
   `source_tap_witness_index_out_of_range`, …) naming the frame/reading, never a
   silent absence and never a fabricated empty witness. A sampler budget
   exhaustion (`WitnessRefusalV1`, `lattice_planes.py:1770`) is recorded as its
   own typed reason.

**No evaluator input may participate in (1)–(7).** Phase A reads no reference
truth, camera manifest, bounds, evaluator mask or JAR; the witness source class
is the screenshot input, not the evaluator renderer. Appearance correction the
pipeline itself applies (for example light-cast/exposure) happens inside inference
and does not change the fact that the witness records the screenshot pixels the
*inference* sampler consumed; the record names the applied correction's digest.

### 0.3 Phase B — project evaluator ownership into the frozen source space

Phase B is a **separate, post-freeze, evaluation-only** run. It must:

1. Re-verify the sealed Phase A digest, the ingress byte-hash/length and
   `FrameTransform` pins, and the image/usable-domain pins byte-identical; on
   mismatch refuse with `usable_domain_pin_mismatch` / `source_pin_mismatch` /
   `byte_digest_mismatch` / `frame_transform_unavailable` **before** any ownership
   lookup, and never recompute the domain from the evaluator side.
2. Load the **separately pinned** evaluator geometry, camera and ownership map
   (`RenderedTruthEvidence`, `evaluation_frozen_planes.py:84`;
   `RendererOwnershipWitnessV1`, `export/attribution.py:74`), which are rendered
   into screenshot coordinates (`render_truth_evidence`, `:1046`).
3. Prove alignment before using anything: evaluator frame height/width equal to
   the Phase A consumed shape and the `FrameTransform` output size; the evaluator
   `CameraModel` intrinsics/pose and the inference camera/intrinsics/pose agree to
   a pinned tolerance (or are the same pinned camera); occlusion/usable convention
   is the pinned evaluator one. If alignment cannot be proven, return the typed
   `source_projection_unavailable` / `evaluator_camera_misalignment` /
   `dimension_mismatch` / `transform_mismatch` / `occlusion_unavailable`
   abstention. **Never re-render ownership under a guessed camera.**
4. Map each saved source-pixel index (honoring the transform) into the evaluator
   ownership grid: `owner_id = ownership.reshape(-1)[index]`, `usable` at the same
   index, then `owner_rows[owner_id].physical_plane_key`. Index `-1` is the
   border sentinel; out-of-range/unusable/background/unresolved owners are their
   own typed reasons.
5. Aggregate per cell exactly as `_cell_owners` does
   (`evaluation_surface_support.py:292-431`): `pure` / `mixed` / `unknown`, with
   reason counts.

Phase B **never** replays `plane_signature` on evaluator RGB, never regenerates
signatures or matches, never substitutes evaluator pixels for screenshot
evidence, and never lets an ownership statistic alter Phase A membership,
disposition, signatures or the solve. The join key is the saved witness index
and its digests, not a re-sampled signature comparison.

### 0.4 What the corrected join is, and is not

A valid Phase B join is **renderer-conditioned appearance/visibility evidence**
about the evaluator's model sampled at the exact screenshot pixels the inference
signature used. It is **not** feature/physical correspondence truth, not proof
that a candidate plane is the physical plane, and not a production decision. §6
keeps the strict abstention and proxy limits. §0.5 and §0.6 name the two R3
implementation seams; §3 (M1–M6) adds the capture schema, §4 adds the testable
contracts (C10–C14), §5 rewrites the freeze-then-join protocol, and §8 gives the
single corrected implementation task.

### 0.5 Normative pre-filter matcher-witness seam (R3)

The failure under study includes candidates dropped by `_apply_same_axis_filter`,
so provenance that publishes matcher witnesses only for post-filter survivors
cannot say whether a dropped alternative had image support. The exact boundary is
inside `match_plane_graph` (`plane_graph.py:964`):

1. **Mint.** For every admitted candidate (every `match_plane_pair` result that
   becomes a `PlaneMatch`), the per-candidate witness list is attached at
   admission (`plane_graph.py:1036-1043`) and the single `MatcherPairWitnessV1`
   or `lattice.WitnessRefusalV1` is retained in the local
   `candidate_witnesses[id(match)]` table (`:1072-1076`). Keying is by the
   `PlaneMatch` object identity, which `_apply_same_axis_filter` preserves: it
   returns references to the same objects and never copies them (`:1259-1273`).
2. **Immutable row order.** `matches.sort(key=_match_order)` (`:1078`, order
   `_match_order` `:1136-1139`) fixes the canonical `admitted_row_index` as the
   position in that sorted list.
3. **Disposition.** `_apply_same_axis_filter` (`:1207-1273`) writes, for every
   position, `SAME_AXIS_DISPOSITION_KEPT` / `SAME_AXIS_DISPOSITION_DROPPED`
   (`:1260-1272`) into the parallel `dispositions` list, and
   `SameAxisCandidateObservation(tuple(matches), tuple(dispositions))`
   (`:1083-1085`, dataclass `:298-331`) already carries the complete pre-filter
   set plus dispositions in that same row order.
4. **Publication gap.** The only publication loop (`:1098-1132`) iterates
   `filtered` (the kept list), so a dropped row's already-minted witness in
   `candidate_witnesses` is never emitted. The publication point must become a
   full-universe pass over `enumerate(matches)` (equivalently, over the
   observation's `candidates`/`dispositions` pairs) that emits for **every** row
   `r`:

   - `admitted_row_index = r`;
   - `disposition` (kept/dropped) and `post_match_index` (the index in `filtered`
     when kept, a typed absence when dropped);
   - the row's own `GraphMatcherWitnessEnvelopeV1` + retrievable
     `MatcherPairWitnessV1` (`:1119-1132`) when the witness exists, or the row's
     own typed `WitnessRefusalV1`/`matcher_witness_unavailable` refusal when it
     does not;
   - `match_sha256` over `_indexed_match_payload(match)` (`:1118`, `:459`) and
     `envelope_sha256`, plus both endpoint `signature_sha256` values from
     `signature_sha256_by_identity` (`image_derived_graph.py:5031-5032`).

   A dropped row may **never** be silently absent, and a dropped row may
   **never** reuse a survivor's witness. The row index and disposition must be
   identical whether read from this publication or from
   `SameAxisCandidateObservation`; the provenance record binds both views by the
   same `admitted_row_index`.

5. **Capture wiring.** In `_derive_image_derived_graph`, the pre-filter
   observation is currently attached only when `same_axis_candidate_sink` is set
   (`image_derived_graph.py:5023-5024`) and the witness payload sink only when
   `matcher_witness_payload_sink` is set (`:5025-5032`). The candidate-support
   provenance sink must request **both** in the same pass, so witness and
   disposition are co-collected for the same sorted row universe.

This seam is evaluation-only and default-off: on the default call no
`pair_witnesses` list is allocated (`:1036`), `candidate_observation_sink` is
`None`, and the returned match tuple and graph digest are unchanged. The seam
must not re-run the matcher, re-correlate, re-signature or re-solve.

### 0.6 Normative Phase A single-read ingress chain (R3; full chain closed in R4)

Phase A's screenshot byte hash, byte length and `FrameTransform` cannot be
recovered after decode, and the ordinary ingress currently performs **two
separate physical reads** of every input screenshot before any signature exists:

- **Read #1 — discovery hash.** `discover_media` (`input/media.py:42-81`) calls
  `_sha256(path)` (`:60`, `_sha256` `:84-89`) to read the entire encoded file and
  compute `SourceFrame.content_hash = _sha256(path)` (`:24`) for de-duplication
  (`seen_hashes`, `:56,61-63`); no byte length is retained and the bytes are then
  discarded.
- **Read #2 — decode.** `input.image.load_image` (`input/image.py:14-29`) opens
  `source_path` **again** (`Image.open(source)`, `:25`), applies EXIF orientation
  and `convert("RGB")`, and returns only the decoded array; the encoded bytes are
  never retained.
- The entry loop `image = load_image(frame.source_path)`
  (`image_derived_entry.py:448-455`) therefore decodes from a different physical
  read than the one that produced `content_hash`, and then drops `content_hash`
  entirely — only the source label survives into
  `build_image_derived_session(frames, source_paths=labels)` (`:458`).
- `build_image_derived_session` (`image_derived_session.py:280-316`) binds only
  the decoded pixels and `image_sha256(pixels)` (`:315`, `:264`); the frame
  record (`ImageDerivedFrame`, `:124-138`; `_FRAME_KEYS`, `:88`;
  `_frame_payload`, `:685-691`) carries `frame_id`, `source_label`,
  `image_shape` and `image_sha256` only. No byte hash/length and no
  `FrameTransform` (`contracts.py:12-38`) is anywhere on the record, and the
  session digest (`:248-256`) is computed over exactly those fields.
- `derive_image_derived_signatures` (`image_derived_signatures.py:346`) consumes
  the decoded array at `:386-394` and calls `plane_signature` at `:410-413`; the
  transform is never constructed on this path, so it is implicitly identity and
  is not bound to the signature it produced.

Capturing bytes only inside `load_image` would still leave two reads and a
discovery hash disconnected from the decoded pixels. Phase A must therefore make
the **whole ingress chain single-read** and add a typed ingress provenance record
threaded to the signature call:

1. **Single-read ingress chain (mandatory; R4 closure).** For each input
   screenshot the encoded bytes must be read from disk exactly **once** into one
   bounded buffer, and that same buffer must serve both:
   - (a) the content hash, byte length and de-duplication that today come from
     `discover_media._sha256(path)` (`input/media.py:60,84-89`, `seen_hashes`
     `:56,61-63`) and populate `SourceFrame.content_hash` (`:24`); and
   - (b) the decode that today comes from the second `Image.open(source)` in
     `input.image.load_image` (`input/image.py:25`) and yields the RGB array.

   Two implementations are admissible and must be equivalent under this
   contract. **(A) Shared buffer:** discovery hashes the bytes it has already
   read and the decoder opens those same bytes (for example
   `Image.open(BytesIO(buffer))`), so the separate `_sha256(path)` and the second
   `open(source)` both disappear. **(B) Discovery-read removal:** `discover_media`
   performs no read at all, `SourceFrame.content_hash`/byte length are derived
   from the ingress record produced by the one decode-feeding read, and
   de-duplication is performed on that same record/hash. In both cases
   de-duplication (duplicate byte content still collapses onto the first path in
   stable order), stable path order, the decoded pixel array and the existing
   `session_sha256` on the default no-ingress path must be unchanged. **No later
   step may call `_sha256(source_path)` again, re-open `source_path`, or
   otherwise re-read the file.** The buffer is transient and bounded; it is not
   serialized into the ingress record or the session.
2. **Ingress record.** A typed record (for example
   `mc-photogrammetry-frame-ingress-provenance-v1`) binds, per frame: `frame_id`;
   `source_label` (path as a label only); `byte_sha256` and `byte_length`
   computed from the shared read/buffer; `decoded_shape`/`decoded_dtype`;
   `decoded_pixel_sha256` (the existing `image_sha256` domain); the reversible
   `FrameTransform` from screenshot source coordinates to consumed coordinates;
   and the decoder provenance (`load_image`, EXIF-transpose + RGB). It must be
   constructed at the ingress site (`image_derived_entry.py:448-455`) before
   `build_image_derived_session`, from the same read that fed the decode.
3. **Threaded binding.** The ingress record must be carried by the same
   `ImageDerivedFrame`/session that carries the decoded pixels, so the session
   digest binds byte hash + pixel digest + transform together; it must survive
   `image_derived_session_to_record`/`from_record` and the pixel-free replay, and
   it must reach `derive_image_derived_signatures` so the source-tap witness
   (§0.2 item 6, §3 M4) is bound to the ingress record that produced the decoded
   frame. Because extending the session record changes `session_sha256`, the
   binding is additive and version-gated: the default entry that supplies no
   ingress record must be byte-for-byte output-neutral.
4. **Typed refusal.** If the observer path is enabled but any ingress pin is
   unavailable or inconsistent — missing encoded bytes/byte digest/byte length,
   no `FrameTransform`, transform/decode shape disagreement, a byte hash that
   does not match the decoded-source bytes, a content hash and decoded pixels
   that cannot be shown to come from the same single read/buffer, or a
   session/record that cannot carry the pin — Phase A refuses with
   `ingress_provenance_unavailable` / `byte_digest_mismatch` /
   `frame_transform_unavailable` naming the frame. It must never fall back to a
   separate `_sha256(source_path)` verification read, a re-open of
   `source_path`, defaulting the transform to identity, or omitting the pin
   silently.

No evaluator input participates in ingress: the single encoded-byte read, decode,
transform and pixel digest are the screenshot ingression only.

## 1. Source-function/field trace

### 1.1 Ordinary image-derived candidate plane creation

The production "ordinary" path is pixel-derived, not keypoint-derived:

1. **Region proposal (image pixels).**
   `region_growth.propose_image_planes_v2`
   (`src/mc_photogrammetry/reconstruction/region_growth.py:859`) returns
   `PlaneProposalResult` (`boundary_candidates.py:152`) holding `PlaneProposal`
   rows (`boundary_candidates.py:133`): `axis` (missing Manhattan direction),
   `side` (sign along the calibrated camera ray), `polygon` (image-pixel
   corners), `mask` (the pixel support, read-only), `strength`, `sources`,
   `coverage`.
2. **Lattice reader.**
   `lattice_planes.frame_plane_readings_from_proposals`
   (`lattice_planes.py:1568`) calls `_proposal_mask_candidates`
   (`lattice_planes.py:1445`), which intersects every proposal mask with the
   caller usable mask, recomputes the geometric side (proposal side metadata is
   ignored), and builds `PlaneRegionEvidence` (`lattice_planes.py:148`) through
   `_make_region_evidence`: exact packed little-endian mask, `bbox_xyxy`,
   `image_shape`, `sources`, and a content `source_region_id` (64-hex).
   `_measure_plane_candidates` (`lattice_planes.py:1304`) then measures the
   window, period/phase, distance alternatives and structural share and builds
   one `PlaneReading` (`lattice_planes.py:356`).
3. **`PlaneReading` support-carrying fields** (all image-derived):
   `polygon` (image pixel polygon), `region_evidence` (`PlaneRegionEvidence`:
   exact packed mask, bbox, image shape, `source_region_id`), `rectification`
   (`PlaneRectification`, `lattice_planes.py:131`: `axis_matrix`, `focal`,
   `principal_point`, `centre`), `alternatives` (`DistanceAlternative`,
   `lattice_planes.py:116`), `period_pixels`, `phase_pixels`, `extent`,
   `samples`, `coverage_cells`, `structural_share`, `confidence`,
   `phase_evidence`.
4. **Canonical index.** `reading_order_key` (`lattice_planes.py:1911`) orders
   each frame's readings content-only (including `polygon` and
   `region_evidence.source_region_id`); `_canonical_reading_index`
   (`plane_graph.py:123`) turns that into the `(frame, plane_index)` keys the
   matcher and solver share.

**Keypoint/descriptor evidence** (`pipeline/correspondence.py`,
`joint_matching.py`, `directed_matching.py`) feeds the accepted camera/pose
chain and `ImageDerivedConnectionGraph`. It reaches the candidate graph only as
relative yaw classes, projected into `quadrant_constraints` in
`_connection_constraints` (`image_derived_graph.py:4871`) via
`_reading_present_constraint_projection` (`image_derived_graph.py:4816`). It
carries no per-plane pixel support into `PlaneMatch`.

### 1.2 Signatures (per reading alternative)

`lattice_planes.plane_signature` (`lattice_planes.py:1664`) builds a
`PlaneSignature` (`lattice_planes.py:462`): `cells`, `interior`, `coverage`,
`origin_uv`, and — for proposal readings — `region_footprint`
(`PlaneRegionFootprint`, `lattice_planes.py:187`) holding
`inside_region_sample_counts`/`valid_sample_counts` per cell, `grid_shape`,
`subsample_factor`, `total_samples_per_cell`.

When and only when a `witness_sink` is passed, `plane_signature` also emits
`SignatureSamplingWitnessV1` (`lattice_planes.py:485`): `cell_rectangles`,
`cell_source_ptr`, `cell_source_indices` (per-cell **source image flat pixel
indices**), plus nested `lattice.RectificationWitnessV1`
(`lattice.py:586`, `map_x/map_y/depth_valid/tap_ptr/tap_source_indices/
tap_weight_numerators`). The ordinary production call
`derive_image_derived_signatures` (`image_derived_signatures.py:346`) passes **no**
witness sink, so production `SignatureEvidence` (`image_derived_signatures.py:124`)
retains only the aggregate arrays; `_signature_payload`
(`image_derived_signatures.py:111`) hashes `frame_id`, `canonical_index`,
`reading_axis`, `alternative_index`, `blocks_per_period`, cells/interior/coverage
and `origin_uv`/`region_footprint` into `signature_sha256`. The sampling witness
is not part of that digest or record.

**R2 correction to the evaluator-replay assumption (normative).** The existing
evaluation binding reconstructs that witness by replaying `plane_signature`
against `truth_evidence.rgb`/`.usable` and requiring exact signature equality
(`evaluation_signature_sampling.py:146-170`; `usable` is required to equal the
production GUI mask at `:140-144`). That makes the witness a property of the
**evaluator render**, not of the screenshot the product consumed, so shader,
lighting, resource-pack and appearance differences can change the signature
arrays/correlation and the join. §0 replaces it: Phase A captures the witness
from the actual screenshot inside the normal signature call, and Phase B projects
evaluator ownership into that frozen source-pixel space. The replay helper above
remains a historical evaluation mechanism and is **not** the source-pixel
witness the corrected design requires.

### 1.3 Matching and `PlaneMatch`

`match_plane_pair` (`plane_graph.py:738`) maps the axis with `axis_transform`
(`plane_graph.py:529`), builds the cell grid homography with
`cell_grid_homography` (`plane_graph.py:543`), warps with `warp_signature`
(`lattice_planes.py:1985`), correlates with `_correlation_surface_with_support`
(`plane_graph.py:626`) and scores (`plane_graph.py:879`):

```
0.5 * min(psr / 6.0, 1.0) + 0.3 * overlap_fraction
+ 0.2 * clip((peak - second_peak)/peak, 0, 1) - 0.5 * (1 - share)
```

`PlaneMatch` (`plane_graph.py:212`) retains: `left`/`right` frame ids,
`left_axis`/`right_axis`, `left_b`/`right_b` (selected alternative period),
`quadrant_of_right`, `psr`, `peak`, `second_peak`, `overlap_cells`,
`integer_residual` (filled later by `solve_component`), `score`,
`left_plane`/`right_plane` (per-frame reading index), `cell_shift_rc`
(**cell-grid** shift, not source pixels).

**Warp orientation and logical-pair order (normative).** `match_plane_pair`
warps the **left signature into the right grid** (`plane_graph.py:804-810`), and
the logical pair order is **left-source cell then right-target cell**
(`plane_graph.py:914-928`). Therefore in every captured record `left` is the
source, `right` is the target, `source_shape`/`target_shape` are the left/right
cell-grid rows×columns, and any warp witness is oriented source-row/column →
target-row/column. This orientation is a deliverable, not an inference, and must
be asserted by asymmetric non-identity tests (§3, §4 C7).

`match_plane_graph` (`plane_graph.py:964`) enumerates frame-pair × quadrant ×
signature-pair candidates, resolves the reading indices, sorts with
`_match_order` (`plane_graph.py:1136`), and applies `_apply_same_axis_filter`
(`plane_graph.py:1207`). That filter groups by
`(left, right, left_axis, quadrant_of_right)`, computes the signed displacement
`_delta` (`plane_graph.py:1142`), drops the lower-scoring member when the
disagreement exceeds `SAME_AXIS_TOLERANCE = 0.15`, and drops both on an exact
score tie (`<= 1e-12`). An optional `candidate_observation_sink` records
`SameAxisCandidateObservation` (`plane_graph.py:298`) = the complete pre-filter
admitted tuple plus a parallel kept/dropped disposition tuple.

`PlaneMatch` therefore retains **no** image polygon, region id, cell footprint,
homography, sampling witness, source pixel coordinate, visibility or ownership.
`cell_shift_rc` and `overlap_cells` describe the correlation *grid*, not pixel
ownership.

**Identity uniqueness is not established.** The shared match identity
`_input_match_identity` (`image_derived_graph.py:4474`) is a 9-scalar tuple
`[left, left_plane, left_axis, right, right_plane, right_axis,
quadrant_of_right, left_b, right_b]`. Nothing in the traced source proves that
this tuple is unique across admitted candidates: two distinct admitted matcher
candidates can coincide on all nine scalars (for example when signature
alternatives or envelope instances differ while the selected `b` and reading
indices agree). It must therefore be treated as a convenience field, never as
the join key (§3 M2).

### 1.4 Solver ingestion

`solve_component` (`plane_graph.py:1308`) and `_evaluate_assignment`
(`plane_graph.py:1405`) read only: reading `frame`, `axis`, `side`,
`alternatives` (via `distance_for`), and match `left`, `right`, `left_plane`,
`right_plane`, `left_axis`, `right_axis`, `left_b`, `right_b`,
`quadrant_of_right`, `score`. This is independently documented by the accepted
replay harness (`scripts/evaluation/replay_same_axis_counterfactual.py:44-56`):
`psr`, `peak`, `second_peak`, `overlap_cells`, `cell_shift_rc`,
`period_pixels`, `phase_pixels`, `extent`, `samples`, `polygon`,
`coverage_cells`, `structural_share`, `confidence`, `rectification`,
`region_evidence` and `phase_evidence` are **never read** by the solver.
`annotate_integer_residuals` (`plane_graph.py:1159`) later fills
`integer_residual` from the solved translations.

### 1.5 Same-axis capture serialization (evaluation-only)

All of these live in `pipeline/image_derived_graph.py` and are built only inside
the failure-observer block of `_derive_image_derived_graph`
(`image_derived_graph.py:4959`, sink block `:5055-5133`); the default call
allocates nothing and returns a byte-identical result:

- `SameAxisCandidateCapture` (`:3299`) — v1 world-axis 1 / v2 all axes. Group
  row (`_validate_same_axis_group_row_v1` `:3457`, v2 `:3487`) is
  `[group_index, left, right, left_axis, relative_quadrant, survivor
  match index/indices, survivor identity/identities]`. Candidate row
  (`_validate_same_axis_candidate_row` `:3543`) is 19 scalars:
  `[group_index, match_index, row_index, left_ref, left_alt, right_ref,
  right_alt, relative_quadrant, score, psr, psr_inf, peak, second_peak,
  overlap_cells, shift_row, shift_col, delta, disposition, reason]`. Reading
  table row (`:2746`, built at `:4342`) is
  `[reading_index, frame, plane, local_axis, side, alternatives]`, alternatives
  `[[b, distance_blocks, support], ...]`. Builder:
  `_collect_same_axis_candidates` (`:4109`).
- `SolverInputBindings` (`:2653`) — every post-filter input, endpoints bound to
  a retained-alternative index and its measured table; edge row 8 columns
  (`:2789`).
- `SolverPhaseCapture` (`:1951`), `FailedActiveEdgeAliases` (`:1657`),
  `SolverSolveContext` (`:666`) — integer-face, active-edge and call-argument
  views respectively.
- The shared match identity is `_input_match_identity` (`:4474`): the 9 scalars
  `[left, left_plane, left_axis, right, right_plane, right_axis,
  quadrant_of_right, left_b, right_b]`. `_match_payload` (`:389`) is the wider
  matcher payload.
- Frozen runner (`reconstruction/frozen_image_runner.py`) `_reading_record`
  (`:139`) serializes polygon and rectification but not `region_evidence`;
  `_match_record` (`:168`) is pinned to the historical field set and omits
  `cell_shift_rc`; signature records (`:486`) store only the three arrays,
  `origin_uv` and `blocks_per_period` — not `region_footprint` and not the
  sampling witness.

**Coverage gap in the current capture (normative).** `candidate_observation_sink`
sees the pre-filter admitted candidates (`SameAxisCandidateObservation`,
`plane_graph.py:298`) and their dispositions, but the matcher witness envelopes
are emitted only by the survivor loop `for match in filtered:` at
`plane_graph.py:1098-1132`. The per-candidate witness itself is already minted
for every admitted match and retained in the local
`candidate_witnesses[id(match)]` table (`:1036-1043`, `:1072-1076`); the dropped
rows' witnesses are simply never published. `SameAxisCandidateCapture` records a
disposition scalar but carries no witness and no endpoint signature for the
dropped members. A design that captures only post-filter survivors cannot answer
whether a dropped alternative was truth-supported, which is exactly the failure
under study. The immutable row index is the position after
`matches.sort(key=_match_order)` (`:1078`), and the disposition is written in
that same order by `_apply_same_axis_filter` (`:1260-1272`); §0.5 makes the
full-universe publication and the row↔disposition↔witness binding mandatory
(§3 M1, §4 C13).

**Lost by capture:** no polygon/footprint/region id, no `signature_sha256`, no
matcher-witness hash, no source-pixel index, no `physical_plane_key`, no
ownership, no retrievable logical-pair/support payload. Every captured field is
an edge/measurement scalar, an alternative table entry, or a typed disposition.

**Source-tap witness partition (normative).** The matcher witness envelopes above
(`GraphMatcherWitnessEnvelopeV1` / `MatcherPairWitnessV1`,
`plane_graph.py:334,393`) are emitted only for post-filter survivors, and the
same-axis capture carries no endpoint signature or sampling witness at all. The
**source-pixel tap witness** (`SignatureSamplingWitnessV1`) that §0 requires is a
different object living at the signature stage, and it too is absent from every
production record because the call passes no `witness_sink`. Both must be
captured: the sampling witness for *where each signature cell's pixels came from
in the screenshot*, and the matcher envelope for *which cells the correlation
paired*. Neither may be reconstructed from evaluator pixels.

### 1.6 Evaluator-only factory renderer / ownership maps and the stable join key

- `export/attribution.py`: `RendererOwnerRowV1` (`:58`) with `owner_id`,
  `block_coordinate`, `slot`, and `physical_plane_key = (axis, sign, numerator,
  denominator)` — the exact rational world face constant, computed in
  `_owner_metadata` (`:111`). `RendererOwnershipWitnessV1` (`:74`) is the ordered
  owner table (`owner_id == row index`).
- `evaluation_frozen_planes.py`: `RenderedTruthEvidence` (`:84`) carries
  `rgb`, `ownership` (per-pixel integer `owner_id`, `-1` = background),
  `owner_index` (`[(block_coordinate, slot), ...]`) and `usable`.
  `bind_renderer_ownership_witness` (`:188`) binds evidence↔witness by
  `evidence_sha256`, `ownership_sha256`, `owner_index_sha256`,
  `renderer_witness_sha256`.
- `evaluation_signature_sampling.py`
  `replay_signature_sampling_witness` (`:122`) replays `plane_signature` with a
  witness sink against the evaluation RGB/usable, requires exact equality of
  cells/interior/coverage/`origin_uv`/`region_footprint`, and emits
  `SignatureSamplingBindingV1` (`:24`) binding `signature_sha256` +
  `truth_evidence_sha256` + `witness_sha256`. **R2 boundary:** this reconstructs
  the witness from evaluator pixels and is therefore not admissible as the Phase
  A source-pixel witness; Phase B must instead project ownership into the frozen
  source indices (§0.3). `SignatureSamplingBindingV1` becomes a Phase B
  *binding-of-projection*, not a signature-equality gate.
- `evaluation_surface_support.py` `_cell_owners` (`:292`) walks each signature
  cell's `cell_source_ptr`/`cell_source_indices`, reads
  `RenderedTruthEvidence.ownership` at those source pixels, and resolves
  `owner_rows[owner_id].physical_plane_key`; a cell is `"pure"` (one plane),
  `"mixed"` or `"unknown"` with typed reason counts. Its index space is exactly
  the flat source-pixel space of the signature it is given, so feeding it the
  **frozen Phase A witness** plus an evaluator ownership map **projected into that
  same space** performs the corrected join without any signature replay.
  `evaluate_surface_support_from_payload` (`:434`) joins
  `SignatureEvidence.signature_sha256` to the sampling payload and to
  `GraphMatcherWitnessPayloadV1` via `left_signature_sha256`/
  `right_signature_sha256`, then uses `MatcherPairWitnessV1.logical_pairs`
  (`plane_graph.py:334`) and `WarpSignatureWitnessV1.source_rows/columns`
  (`lattice_planes.py:1932`) to attribute each correlation pair cell.
- `evaluation_surface_support_streaming.py`
  `summarize_signature_ownership` (`:131`) emits
  `SignatureCellOwnershipSummaryV1` (`:75`) joined by
  `(frame_id, canonical_index, alternative_index, signature_sha256)` with
  per-cell `cell_states` and `physical_planes`.

**Stable join keys (chain):**
`PlaneMatch` → reading via `(left/right, left_plane/right_plane, b)`;
reading → `SignatureEvidence` via `(canonical_index, alternative_index/b)` →
`signature_sha256`; match → signature via
`GraphMatcherWitnessEnvelopeV1.left/right_signature_sha256` (evaluation-only);
signature cell → **screenshot source pixels** via the Phase A
`SignatureSamplingWitnessV1` captured *in the signature call*; source pixel →
physical plane via an evaluator ownership map **projected into that same source
space** (`RenderedTruthEvidence.ownership` + `RendererOwnershipWitnessV1`) under
the §0.2/§0.3 dimension/camera/transform/occlusion pins. The same-axis candidate
capture currently carries **none** of the last four keys, and its 9-scalar
identity is not proven unique (§1.3), so a captured candidate cannot be joined
one-to-one to the sampling/ownership summaries without an additional
matcher-witness capture, the captured source-tap witness, and explicit
match/envelope/signature/alternative-index keys. The evaluation-time signature
replay (`replay_signature_sampling_witness`) is **not** on this corrected chain —
it would make the ownership map depend on evaluator RGB rather than the
screenshot.

### 1.7 Screenshot ingress provenance trace (R3; two-read gap closed in R4)

The pins Phase A needs are created (or discarded) before any signature exists,
and the ordinary chain currently reads each screenshot twice:

- **Discovery (read #1).** `discover_media` (`input/media.py:42-81`) lists image
  paths in stable order and, for every image candidate, reads the entire encoded
  file through `_sha256(path)` (`:60`, `_sha256` `:84-89`) to compute
  `content_hash` for de-duplication (`seen_hashes`, `:56,61-63`), storing it on
  `SourceFrame.content_hash` (`:24`). This is the screenshot byte SHA-256, but no
  byte length is retained, the bytes are discarded, and — critically — this is a
  *different physical read* from the later decode (read #2). The entry reaches it
  through `_discover` (`image_derived_entry.py:541-546`).
- **Entry (read #2).** `run_image_derived_entry` (`image_derived_entry.py:405+`)
  discovers the manifest (`:430`) and, for each image frame, calls
  `load_image(frame.source_path)` (`:450`), storing only the decoded array
  (`:454`) and a path label (`:455`). `frame.content_hash` is not read; the
  manifest's byte hash never reaches the session. `load_image`
  (`input/image.py:14-29`) opens the path a **second** time
  (`Image.open(source)`, `:25`), decodes, and returns the array; the encoded
  bytes are dropped. The file is thus physically read once for discovery and once
  for decode, with no proof that the two reads observed the same bytes (a file
  change in between would split the discovery hash from the decoded pixels).
- **Session.** `build_image_derived_session` (`image_derived_session.py:280-316`)
  captures immutable decoded pixels and `image_sha256(pixels)` (`:309-315`) and
  binds them into `ImageDerivedFrame` / `_frame_payload` (`:124-138`,
  `:685-691`) and the session digest (`:248-256`). There is no byte hash, byte
  length, or `FrameTransform` field.
- **Signature call.** `derive_image_derived_signatures` (`:346`) replays the
  session (`_replay_session`, `:304-318`) and uses `retained_frames` pixel arrays
  (`:386-394`) for `plane_signature` (`:410-413`). The transform is never
  constructed, so no transform binds the decoded frame to the screenshot source
  coordinates.

`FrameTransform` (`contracts.py:12-38`) exists and is carried by
`FrameObservation` (`:41-50`) and consumed by camera adaptation (`camera.py:75`),
but the normal image-derived entry builds no `FrameObservation` and records no
transform. The corrected design closes the full chain: it (a) reads the encoded
bytes exactly **once** into a shared buffer — or removes the discovery read so
`content_hash` is derived from the ingress record — so the discovery content
hash/length/de-duplication and the decode come from the same bytes, (b) builds the
identity (or exact) `FrameTransform` from that same decode, (c) carries all three
through the session record into the signature call, and (d) refuses typed when
they cannot be produced (§0.6, §3 M4, §4 C14). No later path re-hashes or re-opens
`source_path`.

## 2. Missing provenance needed to bind candidates to sampled pixel support

`evaluation_surface_support.py:27-32` already names the gap:
`rectification_nonzero_source_taps`, `gradient_neighbor_source_taps`,
`cell_aggregation_source_taps`, `renderer_subface_plane_identity`, and
`evaluate_surface_support` (`:186-190`) deliberately refuses with
`unavailable_exact_dependencies` rather than substituting whole-reading
aggregates.

The candidate-level gaps are:

1. **Pre-filter coverage, disposition and witness publication.** The
   admitted-candidate universe (including members dropped by
   `_apply_same_axis_filter`) has a disposition in
   `SameAxisCandidateObservation` (`plane_graph.py:298,1207`) but no bound
   witness or typed witness refusal, because the witness publication loop
   (`:1098-1132`) publishes only survivors. This is the primary gap; §0.5/§3 M1
   require a full-universe publication binding `admitted_row_index` (post-sort,
   `:1078`) to `disposition` (`:1260-1272`) and the row's own witness/refusal.
2. **Candidate identity.** Present in `_input_match_identity`, but not carried
   next to pixel support in a capture that also names the signature, and not
   proven unique.
3. **Exact match/envelope identity.** `match_sha256` over `_match_payload`
   (`:389`) and `envelope_sha256` are absent from the same-axis and
   solver-input captures.
4. **Endpoint reading identity.** `(frame, plane, local_axis, alternative_index,
   side, b)` exists in capture reading tables; it does not name the signature,
   and `alternative_index` is not an explicit join column.
5. **Endpoint signature digest.** `signature_sha256` is the only stable key
   into the sampled-pixel and ownership maps; it is absent from the same-axis
   and solver-input captures.
6. **Matcher witness identity.** `envelope_sha256`, `matcher_witness_sha256`,
   `witness_sha256`, `warp_witness_sha256` exist only in the optional matcher
   witness payload sink.
7. **The actual correlation support set.** `MatcherPairWitnessV1.logical_pairs`
   (length must equal `overlap_cells`) plus `WarpSignatureWitnessV1`
   source-pixel rows/columns. These are the exact pixels behind the correlation,
   they are oriented left-source → right-target, and production `PlaneMatch`
   keeps only `overlap_cells` and `cell_shift_rc`. A digest alone does not make
   them retrievable.
8. **Sampled-pixel binding.** `SignatureSamplingBindingV1.report_sha256` /
   `witness_sha256`; absent from every production record. Under R2 this must be a
   *source-screenshot* binding captured in Phase A, not the evaluation-time
   replay binding.
9. **Ownership/visibility aggregate.** Per-candidate same/different/mixed/
   unknown cell counts, as produced by `SurfaceLocalSupportReportV1`; absent by
   design (evaluation-only).
10. **Phase A source-tap witness and screenshot/transform/usable pins (R2,
    primary).** No production record retains the `SignatureSamplingWitnessV1`
    produced by the ordinary signature call, nor the screenshot byte hash,
    consumed dimensions, decode/crop/resize `FrameTransform`, inference-side
    usable-domain digest or sampler settings needed to place those source-pixel
    indices. Without them Phase B cannot map saved indices to ownership in the
    screenshot space, and the existing evaluator replay would silently substitute
    evaluator RGB.
11. **Pinned evaluator projection into the source space (R2).** No record
    binds the evaluator camera/intrinsics/pose, frame dimensions and ownership map
    to the Phase A source coordinate space. The alignment proof (dimension,
    camera, transform, occlusion) and its typed-unavailable refusals are missing,
    so an ownership lookup could otherwise be performed in the wrong space.
12. **Pre-filter matcher-witness publication seam (R3).** No record binds a
    dropped candidate's already-minted witness/refusal
    (`candidate_witnesses[id(match)]`, `plane_graph.py:1036-1043,1072-1076`) to
    its `admitted_row_index`, `disposition`, `match_index`, `match_sha256`,
    `envelope_sha256` and endpoint signature digests. Without it, a dropped
    candidate cannot be attributed to sampled pixel support.
13. **Ingress byte/transform binding (R3; single-read chain R4).** No record binds
    the encoded-file SHA-256/length and the reversible `FrameTransform` to the
    same decoded frame and session whose pixels the signature sampler consumed.
    The byte SHA exists transiently as `SourceFrame.content_hash` but it is
    produced by a discovery read (`input/media.py:60`, `_sha256` `:84-89`) that is
    *separate* from the decode read (`input/image.py:25`) and is then dropped by
    the entry (`image_derived_entry.py:448-455`); the session record
    (`image_derived_session.py:88,685-691`) and the signature call
    (`image_derived_signatures.py:386-413`) carry only decoded pixels. Because the
    hash and the pixels come from two physical reads, there is no proof they saw
    the same bytes. Without a single shared read/buffer (or a discovery that
    derives `content_hash` from the ingress record) plus the byte/transform pin,
    Phase B cannot map frozen source indices to screenshot coordinates and cannot
    detect a substituted/transformed frame.

Note the distinction the source makes explicit: `region_footprint` records
*region-mask occupancy per cell* ("This records coverage only; no cell is
classified occupied", `lattice_planes.py:189-194`). It is not the correlation
support set and not pixel ownership.

## 3. Minimal evaluation-only capture/schema/test changes needed (mandatory)

Additive, default-off, no production behaviour change. M1–M6 below are
mandatory conditions, not options; M4–M6 are the R2 source-screenshot witness
conditions, extended by R3 and closed by R4: M1 names the exact full-universe
matcher-witness publication seam (§0.5), and M4 binds the ingress byte hash/length
and `FrameTransform` captured from **one** shared discovery/decode read/buffer
(§0.6).

**Capture (three seams).** (i) At the ordinary discovery/decode ingress
(`discover_media` `input/media.py:60,84-89`; entry loop
`image_derived_entry.py:448-455`), read each screenshot's encoded bytes exactly
**once** into one bounded buffer shared by the discovery content hash/length/
de-duplication and the decode — or remove the discovery read and derive
`SourceFrame.content_hash`/byte length from the ingress record produced by that
one decode-feeding read — then capture the byte SHA-256/length from those bytes
and the `FrameTransform` from that same decode, carrying them through the session
record into the signature call (R3/R4, §0.6, §3 M4). No second `_sha256(path)` or
`Image.open(source_path)` may occur. (ii) At the signature producer, an optional
`signature_witness_sink` on `derive_image_derived_signatures`
(`image_derived_signatures.py:346`) makes the ordinary call pass a witness sink
into `lattice_planes.plane_signature` and retain the emitted
`SignatureSamplingWitnessV1` beside the accepted `SignatureEvidence`; the default
call passes no sink and allocates/returns exactly as today. (iii) A new bounded
record, e.g. `mc-photogrammetry-candidate-support-provenance-v1`, built by a new
`failed_candidate_support_provenance(readings, matches, signature_witnesses,
matcher_witness_payload, same_axis_observations)` in
`pipeline/image_derived_graph.py`, attached as an optional
`candidate_support_sink` kwarg to `_derive_image_derived_graph` and invoked
inside the existing failure-observer block (`:5055-5133`). It must consume the
already-collected signature witnesses, the **full-universe** matcher witness
payload (B, §0.5) and the `SameAxisCandidateObservation` universe, the same local
readings/matches, and the ingress record; it must not re-match, re-signature,
re-solve or touch a frame. The witness from (ii) and the ingress record from (i)
are carried through to (iii); they are never recomputed at capture time.

**M1 — pre-filter universe coverage (mandatory).** Every candidate admitted
before `_apply_same_axis_filter` must appear exactly once, in admission order,
with:

- `admitted_row_index` — its ordered row index in the post-sort pre-filter order
  (`matches.sort(key=_match_order)` `plane_graph.py:1078`, order `_match_order`
  `:1136-1139`), not the post-filter index;
- `group_index` and the full group key
  `(left, right, left_axis, quadrant_of_right)`;
- `disposition` exactly as `_apply_same_axis_filter` (`:1207-1273`) decided:
  `SAME_AXIS_DISPOSITION_KEPT` (`"kept"`) or `SAME_AXIS_DISPOSITION_DROPPED`
  (`"same_axis_mismatch"`; the lower-score and exact-tie branches both map to
  this single verdict constant, so a separate `dropped_lower_score`/`dropped_tie`
  scalar must not be invented), plus the `delta` (`:1142`) and the
  `SAME_AXIS_TOLERANCE = 0.15` that produced it;
- a **bound witness** for that admitted candidate or an explicit **typed witness
  refusal** (`matcher_witness_unavailable`, `witness_budget_exhausted`,
  `support_payload_over_cap`, ...) naming the candidate row — never a silent
  absence and never a survivor's witness silently reused for a dropped row. The
  witness is the row's own minted object from `candidate_witnesses`
  (`plane_graph.py:1072-1076`), published for dropped rows by the full-universe
  pass over `enumerate(matches)` described in §0.5, not only by the survivor
  loop (`:1098-1132`);
- the post-filter `match_index` when the candidate was kept (and a typed
  absence — not `-1` aliasing a real index — for dropped rows);
- the **row↔disposition identity guarantee**: the `admitted_row_index` and
  `disposition` recorded here must equal the corresponding position in
  `SameAxisCandidateObservation.candidates`/`.dispositions`
  (`plane_graph.py:1083-1085`). The pre-filter observation and the witness
  payload must be co-collected in the same pass
  (`image_derived_graph.py:5023-5032`).

**M2 — exact unique joins (mandatory).** The 9-scalar `_input_match_identity`
must not be used as a unique key. Each row must carry, in addition to
`input_match_identity` (kept only as a convenience field):

- `admitted_row_index` and, when kept, `match_index`;
- exact `match_sha256` over the canonical `_match_payload` (`:389`);
- exact `envelope_sha256`;
- `left_signature_sha256` and `right_signature_sha256`;
- explicit `left_alternative_index` and `right_alternative_index` (in addition
  to `left_b`/`right_b`), each joined to the reading alternative table by
  `(frame, plane, local_axis, alternative_index)`;
- the shared `failed_assignment_sha256` and `input_match_identity_sha256`
  domains (`:1030`, `:4474`) so the row joins the R7/R9/R10 records.

The join must be one-to-one on
`(admitted_row_index, match_index, match_sha256, envelope_sha256,
left_signature_sha256, right_signature_sha256, left_alternative_index,
right_alternative_index)`. Duplicate 9-scalar identities must be representable
without aliasing, and `b` alone must never resolve a join.

**M3 — orientation and retrievable bounded support payload (mandatory).** Each
row must state and carry:

- orientation: `left` = source, `right` = target; the warp is
  source→target (`plane_graph.py:804-810`); logical pair order is left-source
  cell then right-target cell (`plane_graph.py:914-928`);
- `source_shape` and `target_shape` (left/right cell-grid rows×columns) and the
  warp witness shapes, with equality enforced against the endpoint signatures;
- a **retrievable, bounded** support payload: the
  `MatcherPairWitnessV1.logical_pairs` (`plane_graph.py:334`) for the row plus
  `WarpSignatureWitnessV1.source_rows/source_columns` (`lattice_planes.py:1932`),
  or a bounded retrieval handle whose bytes and cap are recorded and whose
  digests are recomputable. `logical_pairs_sha256` / `warp_witness_sha256` alone
  are insufficient: a digest without retrievable logical-pair/witness data
  cannot support pixel ownership attribution. Over-cap retrieval refuses with
  `support_payload_over_cap` and zero rows for that candidate.

**M4 — Phase A ingress pins + source-tap witness captured in the ordinary
signature call (mandatory, R2; ingress extended R3; chain closed R4).** The
signature producer `derive_image_derived_signatures`
(`image_derived_signatures.py:346,410-413`) must pass a witness sink to
`lattice_planes.plane_signature` for each accepted reading/alternative and retain
the resulting `SignatureSamplingWitnessV1` (`lattice_planes.py:485`). For each
accepted `SignatureEvidence` the record must bind, in the same call that minted
the signature:

- **ingress-bound screenshot identity (§0.6):** the encoded-file SHA-256 and byte
  length captured from the **single shared encoded-byte read/buffer** that also
  feeds discovery's content hash/byte length/de-duplication and the decode (or,
  under the discovery-read-removal variant, the hash derived from the ingress
  record of that one read), co-bound to the decoded shape/`dtype`, the
  decoded-pixel digest (`image_derived_session.image_sha256`), and the decoder
  provenance (`input.image.load_image` `image.py:14`, Pillow
  `exif_transpose`+`convert("RGB")`). The pin must be the hash of the very bytes
  the decoder consumed; it must not be assembled from a discovery-side
  `_sha256(path)` plus a separate `load_image` read, and must not be recomputed
  from `source_path` after decode; unavailable or inconsistent ingress metadata
  refuses typed (§0.6 items 1 and 4);
- the reversible `FrameTransform` (`pipeline/contracts.py:13`) from screenshot
  source coordinates to consumed frame coordinates, constructed from the **same
  single decode of the shared buffer** that produced the bound byte hash and pixel
  digest (identity on the current `image_derived_entry.py:448-458` path, recorded
  exactly if not; a missing transform refuses typed rather than defaulting to
  identity);
- the inference-side usable-mask record consumed
  (`UsableMask`/`masks_sha256`, `image_derived_signatures.py:~90,195-197`),
  including its exact bool bytes and `mask_sha256`;
- `signature_sha256` over the canonical `_signature_payload` (`:111`);
- sampler settings sufficient to replay the tap derivation:
  `max_cells`/`SIGNATURE_PIXELS_PER_CELL`/`MAX_RECTIFIED_SIDE`, resolved
  `sub`/`pixels`/`cells_side`/`band`, interpolation/border/fixed-point encoding
  (`lattice.py:585-665`), `gradient_edge_order`, `subsample_factor`, `seam_band`,
  and the `WitnessBudgetV1` stage limits;
- the witness `cell_source_ptr`/`cell_source_indices` and the nested
  `rgb_rectification`/`coverage_rectification`/`region_rectification` witnesses
  (or a bounded retrievable handle), with `witness_sha256` recomputable;
- the ingress record must be carried by the same `ImageDerivedFrame`/session
  (`image_derived_session.py:124-138,685-691`) and reach this signature call
  (`image_derived_signatures.py:386-413`) so the witness is bound to the ingress
  that produced the decoded frame. Extending the session record is additive and
  version-gated so the default no-ingress path stays byte-for-byte
  output-neutral.

Bindings must be checked, not assumed: `source_shape == consumed_image.shape[:2]`;
every tap/cell index is a valid flat index into that shape or the `-1` border
sentinel; the signature replays from the recorded settings and witness. A failure
is a typed refusal (`source_tap_witness_unavailable`,
`source_tap_witness_budget_exhausted`, `source_tap_witness_shape_mismatch`,
`source_tap_witness_index_out_of_range`) naming frame/reading — **never a silent
absence and never a witness rebuilt from evaluator pixels**. No evaluator truth,
camera, mask or JAR may participate in M4.

**M5 — separate pinning of the evaluator projection into the source space
(mandatory, R2).** The Phase A record carries only image-derived material. The
Phase B evaluator inputs are pinned **separately** to the same source coordinate
space and must not be written into the Phase A freeze:

- evaluator render frame `height`/`width` equal to the consumed shape and the
  `FrameTransform` output size (`render_truth_evidence`, `:1046`);
- the evaluator `CameraModel` intrinsics/pose, recorded and checked against the
  inference camera/intrinsics/pose to a pinned tolerance (or pinned as the same
  camera);
- the evaluator ownership map and owner table
  (`RenderedTruthEvidence.ownership`/`owner_index`, `:84`;
  `RendererOwnershipWitnessV1`, `attribution.py:74`) bound by
  `bind_renderer_ownership_witness` (`:188`);
- the occlusion/usable convention (`create_gui_mask`, `:1117`) pinned
  independently of the Phase A usable domain, so a mismatch is detectable rather
  than assumed away.

If any of these cannot be proven equal/aligned, Phase B returns the typed
abstention (`source_projection_unavailable`, `evaluator_camera_misalignment`,
`dimension_mismatch`, `transform_mismatch`, `occlusion_unavailable`) and
attributes nothing. It must never re-render ownership under a guessed camera.

**M6 — Phase B joins saved indices to projected ownership; no evaluator RGB
(mandatory, R2).** Each saved `cell_source_indices`/`tap_source_indices` entry is
mapped (through the recorded transform) into the projected evaluator ownership
grid and read with `_cell_owners` (`evaluation_surface_support.py:292`) to yield
`pure`/`mixed`/`unknown` per cell with typed reasons. Phase B must **not** call
`plane_signature` on evaluator RGB, must not require signature equality against
an evaluator re-render (`evaluation_signature_sampling.py:146-170` is not on the
corrected path), must not substitute evaluator pixels for screenshot evidence,
and must not feed any ownership statistic back into Phase A or inference. The
only admissible equality gate is that the frozen witness and its digests replay
from the sealed Phase A record.

**Schema.** Complete/refused status, explicit `basis`, digest over the strict
canonical payload, bounded caps reusing the existing
`SAME_AXIS_CANDIDATES_MAX_*` / `SOLVER_INPUT_BINDINGS_MAX_*` values plus a
witness byte/membership budget, and the existing atomic-refusal pattern
(`:3382`, `:2707`). Capture **join keys plus the Phase A source-tap witness (or a
bounded retrievable handle) plus a bounded retrievable matcher-support payload**;
Phase B uses the frozen witness with the projected evaluator ownership map and
reuses `_cell_owners`, so full per-pixel rectified arrays stay out of the record.
The record must not require a later evaluator signature replay to obtain source
indices.

**Tests** (synthetic fixtures only; no image/truth reads):

(a) default call returns an identical `result_sha256` and never invokes the
sink; (b) **pre-filter universe**: every admitted candidate appears exactly once
in admission order, the row count equals the `SameAxisCandidateObservation`
count, and every kept/dropped disposition matches `_apply_same_axis_filter`,
including a dropped alternative carrying a bound witness or a typed witness
refusal; (c) **duplicate 9-scalar identity control**: two admitted candidates
sharing all nine scalars still produce two distinct rows, resolved only by row
index plus `match_sha256`/`envelope_sha256`/signature/alternative-index keys,
and no join resolves via `b`; (d) **one-to-one join**: no row aliases another
and a repeated key refuses atomically; (e) each row's identity field equals
`_input_match_identity` of the match at that index when kept; (f) each row's
endpoint `signature_sha256` equals the corresponding
`SignatureEvidence.signature_sha256` selected by `(frame, canonical_index,
alternative_index/b)`; (g) **asymmetric non-identity orientation**: with
`source_shape != target_shape` and a non-identity warp, the record's orientation
and shapes match left→right; swapping the endpoints changes the recorded
orientation/shapes and must fail the asymmetric expectation (a symmetric
identity case is not acceptable coverage); (h) **support conservation**:
`len(logical_pairs) == overlap_cells == envelope.selected_support`, every pair
index is inside both signature grids, and warp witness shapes equal the endpoint
signature shapes; (i) **retrievable payload**: the payload is retrievable under
cap and every digest is recomputable; over-cap or missing payload refuses with
its typed reason and zero rows; (j) tamper/omission/over-cap refuses atomically
with zero partial rows; (k) the record's shared identity digest equals the
`SolverInputBindings`/`SameAxisCandidateCapture` digest for the same match
sequence.

R2 source-screenshot tests (synthetic fixtures only; no image/truth reads):

(l) **source-tap witness captured in the call**: with the signature witness sink
enabled, each accepted `SignatureEvidence` is accompanied by exactly one
`SignatureSamplingWitnessV1` whose `source_shape` equals the consumed image
shape and whose every `cell_source_indices`/`tap_source_indices` entry is a valid
flat index into that shape (or `-1`), and whose `witness_sha256` replays; the
default no-sink call allocates no witness and returns an identical
`signatures_sha256`; (m) **screenshot/transform/usable binding**: the record's
screenshot digest, consumed shape, `FrameTransform` and usable-mask
`mask_sha256`/`masks_sha256` equal the values the signature call actually used,
and a silently substituted mask/transform/shape refuses
(`source_tap_witness_shape_mismatch` / `usable_domain_pin_mismatch`); a non-
identity crop/resize transform round-trips through `FrameTransform.scale` but a
mismatched one refuses; (n) **sampler-settings binding**: a witness produced
under different `subsample_factor`/`seam_band`/`max_cells`/`WitnessBudgetV1`
fails to rebind to the recorded signature; a budget-exhausted sampler publishes
`WitnessRefusalV1` and the record refuses with
`source_tap_witness_budget_exhausted`, never an empty witness; (o) **no evaluator
RGB in Phase A**: with the evaluator/truth modules absent or poisoned (raising on
import/attribute), the Phase A capture still produces the witness, and no
`truth_evidence_sha256`/`RenderedTruthEvidence`/`physical_plane_key` appears in
the Phase A record; (p) **Phase B projects ownership, not signatures**: given a
frozen Phase A witness and a synthetic evaluator ownership map projected into the
same source shape, per-cell ownership equals a direct read of
`ownership.reshape(-1)[saved_index]`; a poisoned evaluator RGB that would change a
signature replay does **not** change the attribution, and Phase B performs zero
`plane_signature` calls on evaluator pixels; (q) **alignment refusal**: a
dimension/camera/transform/occlusion mismatch between the pinned evaluator inputs
and the frozen Phase A pins returns the typed `source_projection_unavailable` /
`evaluator_camera_misalignment` / `dimension_mismatch` / `transform_mismatch` /
`occlusion_unavailable` abstention with no attribution and no re-render under a
guessed camera.

R3 pre-filter-witness and ingress tests (synthetic fixtures only; no image/truth
reads):

(r) **dropped-candidate witness publication**: in a synthetic pre-filter group
where `_apply_same_axis_filter` drops a candidate, the record contains a row for
that dropped candidate carrying its **own** bound matcher witness or its **own**
typed witness refusal plus `admitted_row_index` and `disposition`; the row is not
omitted, does not carry the survivor's witness/`envelope_sha256`, and a
budget-exhausted dropped candidate carries its own
`WitnessRefusalV1`/`matcher_witness_unavailable` refusal rather than an empty or
reused witness. The default no-sink call publishes nothing and returns the
byte-identical match tuple;

(s) **row↔disposition identity**: for a fixture with both kept and dropped rows,
every `admitted_row_index`/`disposition` pair in the provenance record equals the
position in `SameAxisCandidateObservation.candidates`/`.dispositions`; every kept
row's `match_index` indexes the same match object in the post-filter match tuple;
every dropped row's `match_index` is a typed absence (never an aliased real
index); and tampering a row's disposition, duplicating a row index, or reordering
rows without renumbering refuses atomically with zero partial rows;

(t) **ingress-to-session and ingress-to-signature binding (single shared read)**:
with the ingress capture enabled, each frame's record carries the encoded byte
SHA-256/length, the decoded-pixel digest and the `FrameTransform` bound to the
**same** `ImageDerivedFrame`/session whose decoded array reaches the signature
call, and the byte hash and the decoded pixels are provably from the **same**
single encoded-byte read/buffer (the discovery content hash/de-duplication and the
decode consume the same bytes); mutating the source bytes after the single read,
substituting a different decode, or supplying a transform whose
`source_size`/`resolved_output_size` disagrees with the decoded shape makes the
capture refuse with `byte_digest_mismatch` / `frame_transform_unavailable` /
`ingress_provenance_unavailable`, and the witness is never bound to a different
frame than the bytes it claims; a same-bytes/same-decode run is deterministic;

(u) **typed ingress refusal and output neutrality**: enabling the observer ingress
but withholding the byte hash/length or the transform refuses typed
(`ingress_provenance_unavailable`) naming the frame rather than re-hashing
`source_path`, re-opening `source_path`, defaulting to an identity transform, or
dropping the pin silently; with the observer disabled the ordinary entry returns a
byte-identical result and `session_sha256`, and the session record carries no
ingress fields;

(v) **single-read ingress chain (R4)**: with read-count instrumentation over a
synthetic fixture whose bytes are known, the ordinary ingress reads each input
screenshot from disk exactly **once** — there is no discovery `_sha256(path)`
followed by a second `Image.open(source_path)` — and that one buffer yields both
the `byte_sha256`/`byte_length` used for `SourceFrame.content_hash`/
de-duplication and the decoded array whose `image_sha256` is bound to the session;
rewriting the file content after the single read changes neither the recorded byte
hash nor the decoded pixels (there is no second-read window); two paths with
identical encoded bytes are still de-duplicated by the shared hash in stable path
order; a chain that cannot prove the content hash and the decode share one
read/buffer refuses `ingress_provenance_unavailable`/`byte_digest_mismatch` rather
than silently reading twice; and the default no-ingress entry stays byte-for-byte
output-neutral.

## 4. Testable contracts and failure/abstention rules

- **C1 output neutrality.** With no sink, `_derive_image_derived_graph` returns
  the same `ImageDerivedGraphResult` and `result_sha256`; no list is mutated.
- **C2 pre-filter atomicity.** A complete record covers **every** pre-filter
  admitted candidate exactly once (kept and dropped), not just post-filter
  survivors; a refused record carries exactly one bounded `reason`/`field` and
  zero rows.
- **C3 exact unique join.** Row joins are one-to-one on ordered
  `admitted_row_index` plus `match_sha256`, `envelope_sha256`, both endpoint
  `signature_sha256`, and both explicit `alternative_index` values. The
  9-scalar identity is a convenience field; a test with duplicate 9-scalar
  identities must still resolve two distinct rows. `b` alone never joins.
- **C4 pre-filter disposition.** Every admitted candidate's `disposition` and
  `delta`/tolerance match `_apply_same_axis_filter`, and a dropped candidate has
  either a bound witness or a typed witness refusal naming its row.
- **C5 support conservation.** `len(logical_pairs) == overlap_cells` and each
  pair maps into both signatures' cell grids; `source_shape`/`target_shape`
  equal the endpoint signature grids and the warp witness shapes.
- **C6 signature join.** Each endpoint's `signature_sha256` equals the
  `SignatureEvidence` selected by `(frame, canonical_index,
  alternative_index/b)` and the envelope's hash.
- **C7 warp orientation.** The record states left-source/right-target, the warp
  is source→target, and an asymmetric non-identity test
  (`source_shape != target_shape`, non-identity warp) passes; endpoint swap
  changes the result.
- **C8 retrievable bounded support.** The support payload is retrievable under
  its recorded cap and every digest is recomputable; a digest-only record is
  non-conforming; over-cap retrieval refuses with `support_payload_over_cap`
  and zero rows.
- **C9 replay.** The validator recomputes every digest from the payload and
  refuses on mismatch or over-budget (digest mismatch raises; malformed input
  refuses atomically).
- **C10 source-tap witness in the signature call (R2).** Each accepted
  `SignatureEvidence` is accompanied by exactly one bound
  `SignatureSamplingWitnessV1` produced by the same `plane_signature` invocation,
  with `source_shape` equal to the consumed image shape, every tap/cell index
  valid for that shape (or `-1`), and a recomputable `witness_sha256`; the
  default no-sink path allocates no witness and leaves `signatures_sha256`
  unchanged. A missing/budget-exhausted witness refuses with its typed reason and
  is never silently empty.
- **C11 screenshot/transform/usable pin (R2).** The record's screenshot SHA-256
  and byte length, decoded shape/`dtype`/pixel digest, `FrameTransform`, and the
  inference usable-mask `mask_sha256`/`masks_sha256` equal what the signature
  call consumed; any substitution refuses (`source_pin_mismatch` /
  `usable_domain_pin_mismatch` / `source_tap_witness_shape_mismatch`).
- **C12 evaluator-projection join, no evaluator RGB (R2).** Phase B maps saved
  source-pixel indices into the **separately pinned** evaluator ownership map
  projected into exactly the frozen source space and returns per-cell
  `pure`/`mixed`/`unknown`; it performs no `plane_signature` call on evaluator
  pixels and no signature-equality gate, and no ownership result changes Phase A
  membership, disposition or the solve. If dimension/camera/transform/occlusion
  alignment cannot be proven, it returns the typed abstention and attributes
  nothing.
- **C13 pre-filter witness seam (R3).** Every pre-filter admitted row — kept or
  dropped — carries its own bound matcher witness or its own typed witness
  refusal, published by the full-universe pass (§0.5), with
  `admitted_row_index` = the position after `matches.sort(key=_match_order)`
  (`plane_graph.py:1078`) and `disposition` identical to
  `SameAxisCandidateObservation.dispositions` at that position
  (`:1083-1085,1260-1272`). A dropped row's witness is never omitted and never
  aliases a survivor's; its `match_index` is a typed absence. The default path
  still publishes nothing and returns the unchanged match tuple.
- **C14 ingress pin binding (R3; single-read chain R4).** The encoded-file
  SHA-256/length, the decoded-pixel digest and the reversible `FrameTransform` are
  captured from one **shared** encoded-byte read/buffer that also feeds the
  discovery content hash/byte length/de-duplication (`input/media.py:60,84-89`)
  and the decode (`input/image.py:25`; entry loop `image_derived_entry.py:448-455`),
  and are bound to the same `ImageDerivedFrame`/session
  (`image_derived_session.py:124-138,685-691`) and the same signature call
  (`image_derived_signatures.py:410-413`). There is exactly one physical read per
  input screenshot: either discovery and decode share the buffer, or discovery
  performs no read and `content_hash`/length are derived from the ingress record;
  no path may re-hash or re-open `source_path`. Byte hash, pixel digest and
  transform are mutually consistent (`source_size`/`resolved_output_size` equal the
  decoded shape); a missing or inconsistent pin refuses typed
  (`ingress_provenance_unavailable`, `byte_digest_mismatch`,
  `frame_transform_unavailable`) and is never recovered by a second
  `_sha256(source_path)` read, reopening `source_path`, defaulting to identity, or
  silently omitting the pin. The default no-ingress entry is byte-for-byte
  output-neutral and preserves de-duplication.

**Failure rules (mirror the existing evaluator discipline).** Missing matcher
witness → `matcher_witness_unavailable`; missing/over-cap support payload →
`support_payload_unavailable` / `support_payload_over_cap`; malformed field →
bounded `invalid_*_diagnostic`; over cap → `budget_exhausted`; missing Phase A
source-tap witness → `source_tap_witness_unavailable` (never re-derived from
evaluator pixels and never substituted with a whole-reading aggregate); sampler
budget exhaustion → `source_tap_witness_budget_exhausted`; shape/index
disagreement → `source_tap_witness_shape_mismatch` /
`source_tap_witness_index_out_of_range`; screenshot/transform/usable
substitution → `source_pin_mismatch` / `usable_domain_pin_mismatch`; missing or
inconsistent ingress byte hash/length or `FrameTransform`, or a content hash and
decoded pixels that cannot be shown to share one single read/buffer (R3/R4) →
`ingress_provenance_unavailable` / `byte_digest_mismatch` /
`frame_transform_unavailable`; unprovable
Phase B alignment → `source_projection_unavailable` /
`evaluator_camera_misalignment` / `dimension_mismatch` / `transform_mismatch` /
`occlusion_unavailable`; unusable or unresolved source pixels →
`unknown`/`unresolved_physical_plane_identity`, never a defaulted owner; no
candidate is ever given a fabricated pair, a fabricated witness, a re-sampled
signature, or a default owner. **No Phase A record may contain a truth hash,
ownership value, `physical_plane_key`, evaluator camera manifest or JAR value**,
and no evaluator pixel may regenerate a signature or match.

## 5. Strict freeze-then-evaluator-join protocol preventing oracle leakage

- **Phase A (image-only freeze).** Run the ordinary image-only pipeline and
  serialize only image-derived records (readings, signatures, matches, graph
  result, candidate capture, candidate support provenance, the **source-tap
  witnesses captured inside the ordinary signature call**, and the **ingress
  byte/transform pins captured at the ordinary decode**). The candidate support
  provenance must cover **every** pre-filter admitted row — kept and dropped —
  with its own witness/refusal and row↔disposition binding (§0.5, §3 M1). Phase A
  must not read any reference truth, camera manifest, bounds, evaluator mask
  payload or JAR.
- **Phase A pins (mandatory).** The Phase A record must carry, for each frame:
  the encoded-file SHA-256 and byte length captured from the single shared
  encoded-byte read/buffer that also feeds the discovery content
  hash/length/de-duplication and the decode, the decoded shape/`dtype`/pixel
  digest, the decode provenance (`load_image` EXIF-transpose + RGB), the
  `FrameTransform` from screenshot source coordinates to consumed coordinates, the
  caller **usable-domain mask pin** actually consumed (`mask_sha256` /
  `masks_sha256`), the signature digest, the sampler settings, and the
  `SignatureSamplingWitnessV1` (or a bounded retrievable handle) with its
  recomputable `witness_sha256`. The byte SHA-256/length and `FrameTransform` are
  captured at the one shared read/decode (§0.6) and threaded through the session
  record into the signature call; the content hash and the decoded pixels are
  provably the same bytes, and they are never recomputed later from a second
  `source_path` read, defaulted to identity, or re-sampled from evaluator
  material. These pins identify the exact screenshot pixel domain in which
  candidate support was sampled; no later step may recompute, re-decode or
  re-sample them from evaluator material.
- **Freeze.** Write one artifact to a fresh root; record file SHA-256 and the
  record seal; forbid `.partial`; then treat it read-only (the accepted
  R10/R11/R1x pattern already does exactly this).
- **Phase B (evaluator projection join, separate run).** Only after the freeze
  digest is recorded, load the sealed image-only artifact together with the
  **separately pinned** evaluator camera/geometry/ownership inputs and:
  1. re-verify the Phase A freeze seal, ingress byte-hash/length and
     `FrameTransform` pins, image pins, dimensions, transform and usable-domain
     pins byte-identical; any mismatch refuses with `source_pin_mismatch` /
     `byte_digest_mismatch` / `frame_transform_unavailable` /
     `usable_domain_pin_mismatch` **before** any ownership lookup, and nothing is
     recomputed from the evaluator side;
  2. prove the evaluator frame dimensions equal the consumed shape / transform
     output size, the evaluator `CameraModel` intrinsics/pose agree with the
     inference camera/intrinsics/pose to a pinned tolerance (or are the same
     pinned camera), and the occlusion/usable convention matches the pinned
     convention; if not provable → typed `source_projection_unavailable` /
     `evaluator_camera_misalignment` / `dimension_mismatch` /
     `transform_mismatch` / `occlusion_unavailable`;
  3. bind `RenderedTruthEvidence` ↔ `RendererOwnershipWitnessV1` via
     `bind_renderer_ownership_witness` (`:188`);
  4. join candidate rows one-to-one by the C3 key set, retrieve each row's
     **frozen** source-tap witness and bounded matcher-support payload, map each
     saved source-pixel index (through the recorded transform) into the projected
     ownership/usable grids, and classify each cell with `_cell_owners` using the
     recorded left-source → right-target orientation;
  5. only then compute any support/ownership statistic. **No step replays
     `plane_signature` on evaluator RGB, requires evaluator signature equality,
     or substitutes evaluator pixels for screenshot evidence.**
- **Leakage guards.** The image-only records (§1.5) contain no `physical_plane_key`,
  truth hash, ownership or evaluator camera value; the freeze must precede any
  truth load and should be enforced by an explicit preflight refusal if truth
  inputs appear before the freeze digest exists (the ordering discipline of
  `synthetic_targeted_capture_preflight`); the evaluator must not re-run the
  matcher, re-signature or solver and must not feed ownership back into a
  production decision; no reference/JAR may create or reseal a Phase A record,
  and no Phase A record may be regenerated after evaluator inputs are loaded.

## 6. Risks that make pixel ownership an appearance proxy rather than point truth

- `RenderedTruthEvidence.ownership` is a single z-buffered visible-surface owner
  map from an evaluator render. A signature cell's source pixels can be
  occluded, GUI-masked, or on a different face; `_cell_owners` explicitly counts
  `mixed`, `unknown`, `absent_owner_records`, `unusable_source_pixels` and
  `unresolved_physical_plane_identity` (`evaluation_surface_support.py:277-278`).
- `physical_plane_key` resolves only exact single-coordinate axis-aligned faces
  (`attribution.py:119-125`); partial/non-cube/sub-face owners can leave it
  `None` (`evaluation_frozen_planes.py:1754` warns legacy all-model sub-face
  grouping "has no polygon identity in renderer ownership and can misassign
  repeated same-slot offsets").
- Cell → source-pixel mapping is a rectified nearest/bilinear tap set
  (`INTER_LINEAR`, `lattice.py:586`), and a cell's `cells` value is an aggregate
  gradient-energy sum, not a per-pixel correspondence. A "pure" cell can still
  contain a few foreign pixels.
- The correlation is whole-cell phase correlation over a mean-subtracted window;
  `overlap_cells` counts overlapping cells, and `psr` is an *optional scored
  cue* "never treated as ground truth" (`plane_graph.py:19-21`).
- The evaluator scene geometry (JAR/reference palette) may not match the
  produced blocks, so ownership can attribute pixels to planes of a scene the
  image-derived candidate did not necessarily observe.
- Only cells whose source pixels are usable and owner-resolvable can be
  classified; the remainder are `unknown`, so every rate is conditional on that
  subset.
- **Evaluator-pixel substitution (R2, the failure this correction prevents).** A
  witness reconstructed by replaying `plane_signature` on evaluator RGB
  (`evaluation_signature_sampling.py:146-170`) is conditioned on the evaluator's
  shader/lighting/resource-pack render, not on the screenshot the product
  consumed; signature arrays and the correlation can differ, so the join would
  silently measure the evaluator's appearance rather than the candidate's source
  evidence. The corrected design captures the witness from the screenshot and
  keeps evaluator material out of Phase A.
- **Coordinate-space brittleness (R2).** Saved source-pixel indices are valid
  only inside the frozen decode/transform/usable domain. A crop, resize, EXIF
  orientation change, different GUI mask or different frame dimensions moves the
  indices, so Phase B must refuse (typed-unavailable) rather than guess when the
  pins or the evaluator projection cannot be proven to coincide.
- **One-way ownership (R2).** Even a fully aligned join is evaluator-model
  evidence: it may not feed back to change membership, disposition, signatures or
  the solve, and it cannot upgrade a candidate to point truth.
- **Ingress/digest ordering (R3; chain closed R4).** The encoded-file hash and the
  decode are two separate physical reads on the ordinary path:
  `discover_media._sha256(path)` (`input/media.py:60,84-89`) then
  `load_image`'s `Image.open(source)` (`input/image.py:25`). Binding frozen source
  indices to a byte hash computed from a different read than the decoded pixels is
  valid only if the two are proven to be the same bytes; otherwise a file change
  between discovery and decode would be invisible. §0.6 closes this at the chain
  level: one shared encoded-byte read/buffer feeds both the content
  hash/length/de-duplication and the decode, or discovery performs no read and
  derives `content_hash` from the ingress record, with a typed refusal when a
  single-read proof cannot be shown. A two-read implementation that captures bytes
  only inside `load_image` is non-conforming even if the pins are otherwise
  complete.
- **Dropped-row attribution (R3).** A dropped candidate's witness is evidence
  that the correlation that lost the same-axis filter was still image-supported;
  it is not evidence that the dropped candidate is the physically correct plane.
  Publishing dropped-row witnesses must not be read as filter causality, an
  accuracy gain, or a promotion signal.

**Strict abstention and proxy limits (mandatory).** Where the evaluator cannot
resolve ownership — or cannot prove that its ownership map lies in the frozen
source-pixel space — it must return typed `unknown`/`unresolved_*` /
`source_projection_unavailable` and must never substitute evaluator RGB, a
whole-reading aggregate, a whole-frame rate, or a defaulted owner. No candidate
may be kept, dropped, promoted, or re-scored on the basis of an ownership
statistic; membership and disposition in the frozen artifact are fixed in Phase A
and are not revised in Phase B. A same-surface/different-surface or ownership
verdict is an appearance/visibility proxy conditioned on the evaluator's render,
geometry, camera projection and usable domain; it is evidence about the
evaluator's model and about sampled screenshot pixels, never point truth about
the candidate's physical plane.

**Conclusion.** No accuracy, correspondence-truth or promotion claim follows.
The corrected design produces attributable, retrievable source-pixel provenance
for every admitted candidate plus renderer-conditioned appearance/visibility
evidence, and nothing more.

## 7. Exact source references

| Concern | Source |
|---|---|
| Region proposal → `PlaneProposal` | `region_growth.py:859`; `boundary_candidates.py:133,152` |
| Proposal → reading | `lattice_planes.py:1445,1568,1304`; `PlaneReading:356`; `PlaneRegionEvidence:148` |
| Reading order/index | `lattice_planes.py:1911`; `plane_graph.py:123` |
| Signature + sampling witness | `lattice_planes.py:462,485,1664,187,1985,1932`; `lattice.py:586,655` |
| Production signatures (no witness sink; call site) | `image_derived_signatures.py:111,124,346,410-413` |
| Screenshot decode / transform contract | `input/image.py:14`; `pipeline/contracts.py:13`; `pipeline/image_derived_session.py:264,285-316`; `pipeline/image_derived_entry.py:448-458` |
| Frozen source pixels / signed session | `pipeline/image_derived_session.py:124-159`; `pipeline/image_derived_entry.py:701-739` |
| `PlaneMatch` + score + filter | `plane_graph.py:212,529,543,738,879,964,1136,1142,1207,298` |
| Warp orientation / logical pairs | `plane_graph.py:804-810,914-928,334` |
| Solver reads | `plane_graph.py:1308,1405,1159`; `replay_same_axis_counterfactual.py:44-56` |
| Same-axis capture | `image_derived_graph.py:3299,3457,3487,3543,2746,4109,4342` |
| Other capture records | `image_derived_graph.py:2653,1951,1657,666,389,4474,4959` |
| Frozen serialization | `frozen_image_runner.py:139,168,486` |
| Renderer ownership map | `export/attribution.py:58,74,111`; `evaluation_frozen_planes.py:84,188` |
| Truth render into screenshot coords / usable | `evaluation_frozen_planes.py:1015,1046,1101-1117` |
| Sampling replay/binding (R2: not on corrected path) | `evaluation_signature_sampling.py:24,122-179` |
| Cell→ownership / surface support | `evaluation_surface_support.py:27-32,186-190,292,434`; `evaluation_surface_support_streaming.py:75,131` |
| Matcher witness | `plane_graph.py:334,393,436` |
| Pre-filter witness mint/publication seam (R3) | `plane_graph.py:964,1036-1043,1072-1076,1078,1098-1132,1136-1139,1207-1273,298-331` |
| Pre-filter capture wiring (R3) | `image_derived_graph.py:5019,5023-5032` |
| Screenshot ingress bytes/hash — discovery read (R3) | `input/media.py:24,42-81,84-89` |
| Screenshot decode — second read (R3) | `input/image.py:14-29` |
| Entry decode drops `content_hash` (R3) | `image_derived_entry.py:430,448-458`; `_discover` wrapper `:541-546` |
| Session frame record / digest (R3) | `image_derived_session.py:88,124-138,248-256,280-316,685-691` |
| Signature call consumes decoded frame (R3) | `image_derived_signatures.py:304-318,386-394,410-413` |
| Single-read ingress chain (R4) | `input/media.py:60,84-89` + `input/image.py:25` (must collapse to one read/buffer); `image_derived_entry.py:448-458,541-546` |
| Transform contract / observer (R3) | `pipeline/contracts.py:12-38,41-50`; `reconstruction/camera.py:75` |

## 8. One minimal next implementation task (R4-corrected)

Add the evaluation-only, default-off capture described in §3 at **three** seams,
and nothing else:

**A. Phase A single-read ingress chain at the ordinary discovery/decode (R3, full
chain closed R4).** The ordinary chain currently reads each screenshot twice:
`discover_media` hashes the encoded file via `_sha256(path)`
(`input/media.py:60,84-89`) for de-duplication, then `load_image(frame.source_path)`
re-opens it (`Image.open(source)`, `input/image.py:25`) to decode, and the entry
loop (`image_derived_entry.py:448-455`) drops `content_hash`. Restructure the
ordinary ingress so each input screenshot's encoded bytes are read from disk
exactly **once** into one bounded buffer shared by both stages — the content hash,
byte length and de-duplication are computed from the bytes that were read, and the
decode consumes those same bytes (for example `Image.open(BytesIO(buffer))`) — or,
equivalently, remove the discovery read so `discover_media` performs no
`_sha256(path)` and `SourceFrame.content_hash`/byte length are derived from the
ingress record produced by that one decode-feeding read, with de-duplication
performed on that same hash. In both variants de-duplication (identical bytes
still collapse onto the first path in stable order), stable path order, the decoded
pixel array and the existing `session_sha256` on the default path stay identical,
and **no path calls `_sha256(source_path)` or opens `source_path` a second time**.
Build the reversible `FrameTransform` (`pipeline/contracts.py:13`) from that same
decode. Carry the typed ingress record (byte hash/length, decoded shape/`dtype`,
decoded-pixel digest, transform, decode provenance) through
`build_image_derived_session` (`image_derived_session.py:280-316`) into the session
record and on to `derive_image_derived_signatures`, so the source-tap witness (C)
is bound to the same decoded frame/session. Extend the session/frame record
additively and version-gated so the default no-ingress entry is byte-for-byte
output-neutral; refuse typed (`ingress_provenance_unavailable`,
`byte_digest_mismatch`, `frame_transform_unavailable`) when a pin is unavailable or
inconsistent, including when the content hash and the decoded pixels cannot be
shown to come from the same single read/buffer. Never re-hash or re-open
`source_path` after the read and never default the transform to identity.

**B. Pre-filter matcher-witness publication (R3).** In `match_plane_graph`
(`plane_graph.py:964`), replace the survivor-only publication loop
(`:1098-1132`) with a full-universe pass over the sorted admitted rows
(`enumerate(matches)`, order `:1078`) that, for **every** pre-filter admitted
candidate — kept or dropped by `_apply_same_axis_filter` (`:1207-1273`) — emits
its own `GraphMatcherWitnessEnvelopeV1` + retrievable `MatcherPairWitnessV1`
(from the already-minted `candidate_witnesses[id(match)]`, `:1072-1076`) or its
own typed `WitnessRefusalV1`/`matcher_witness_unavailable` refusal, bound to
`admitted_row_index`, `disposition`, post-filter `match_index` (typed absence
when dropped), `match_sha256`, `envelope_sha256` and both endpoint
`signature_sha256` values. Bind the row index/disposition to
`SameAxisCandidateObservation` (`:1083-1085`) exactly. In
`_derive_image_derived_graph`, request the pre-filter observation and the witness
payload in the same pass (`image_derived_graph.py:5023-5032`). Keep the default
call allocation-free and digest-identical.

**C. Source-tap witness at the ordinary signature call (R2, retained).** Give
`derive_image_derived_signatures` (`image_derived_signatures.py:346`) an optional
`signature_witness_sink`; when enabled, the ordinary `plane_signature` call
(`:410-413`) receives a witness sink and the emitted
`SignatureSamplingWitnessV1` is retained beside its `SignatureEvidence`
(`SignatureSamplingWitnessV1`, `lattice_planes.py:485`). Bind, for each accepted
signature, the ingress record from A (byte SHA-256/length, decoded
shape/`dtype`/pixel digest, transform, decode provenance), the inference
usable-mask `mask_sha256`/`masks_sha256`, `signature_sha256`, the sampler
settings, and the witness with recomputable `witness_sha256`, checking shape and
index validity.

**D. Candidate support provenance record.** Add the `candidate_support_sink`
record in `_derive_image_derived_graph`, consuming the A/B/C captures. For
**every** pre-filter admitted candidate — kept or dropped — record:

1. the ordered `admitted_row_index`, `group_index`, group key, `disposition`,
   `delta`/tolerance, and post-filter `match_index` when kept (typed absence when
   dropped);
2. exact `match_sha256`, `envelope_sha256`, `left_signature_sha256`,
   `right_signature_sha256`, explicit `left_alternative_index` /
   `right_alternative_index`, and the 9-scalar identity only as a convenience
   field;
3. the row's own bound matcher witness or its own typed witness refusal (from B);
4. `overlap_cells`, `cell_shift_rc`, `source_shape`/`target_shape`, the
   left-source → right-target orientation, and a retrievable bounded matcher
   support payload (`MatcherPairWitnessV1.logical_pairs` +
   `WarpSignatureWitnessV1` source rows/columns) with recorded cap and digests;
5. the shared `failed_assignment_sha256` / `input_match_identity_sha256`;
6. the per-frame ingress record from A, bound to the same session.

Implement a strict fail-closed validator with bounded caps and atomic refusal, and
focused tests (a)–(v) in §3: output neutrality, pre-filter universe/disposition
coverage, duplicate-9-scalar-identity uniqueness, one-to-one joins, signature
join, asymmetric non-identity orientation, support conservation, retrievable
payload and over-cap refusal, the R2 tests (l)–(q), and the R3/R4 tests
(r)–(v): dropped-candidate witness/refusal publication, row↔disposition identity,
ingress-to-session and ingress-to-signature binding over one shared read,
typed ingress refusal with a byte-identical default path, and the single-read
ingress chain (exactly one physical read per screenshot, hash + de-dup + decode
from the same buffer, no second `_sha256(path)`/`Image.open(source_path)`).

Preserve the §5 Phase A image-only freeze (now including the single-read ingress
pins and the full pre-filter witness publication) followed by the Phase B
evaluator-projection join, and the §6 strict abstention/proxy limits. No production
behaviour change (default-off) and no solver rerun. The ordinary program ingress
must read each input screenshot's encoded bytes exactly **once** and compute the
byte pins from those exact bytes before/while decoding them, with the discovery
content hash (or de-duplication) taken from that same read (A/§0.6); there must be
no second `_sha256(source_path)`/re-open, no manual or diagnostic re-reads, and no
direct fixture inspection. Continue to forbid **evaluator RGB replay, evaluator
pixels feeding a signature or match**, and any truth/camera/mask/JAR/Minecraft
payload read in this implementation task. §0.5 is the primary contract for B and
§0.6 for A.

## 9. Remediation maps

### 9.1 Luna HOLD map (R1)

| Luna HOLD condition | Where made mandatory |
|---|---|
| 1. Every pre-filter admitted candidate covered with keep/drop disposition and a bound witness or typed witness refusal; post-filter-only coverage insufficient | §1.5 coverage gap; §2.1; §3 M1; §4 C2/C4; tests (b); §8 D.3 |
| 2. No reliance on 9-scalar `_input_match_identity` uniqueness; bind ordered row index + `match_sha256`/`envelope_sha256` + both endpoint signature digests + explicit alternative indices; duplicate-identity and one-to-one join tests; never resolve via `b` | §1.3 uniqueness note; §1.6; §2.2-2.5; §3 M2; §4 C3; tests (c)(d); §8 D.2 |
| 3. Explicit left-source/right-target warp orientation and source/target shapes; asymmetric non-identity tests; retrievable bounded logical-pair/witness payload (digest alone insufficient) | §1.3 orientation; §2.7; §3 M3; §4 C5/C7/C8; tests (g)(h)(i); §8 D.4 |
| Freeze-then-join and proxy limits preserved | §5 (Phase A freeze + image/usable pins, Phase B join); §6 strict abstention |

### 9.2 Luna source-screenshot witness correction map (R2)

| R2 condition | Where made mandatory |
|---|---|
| Phase A captures actual screenshot sampler taps/witness during the normal signature call (not a later evaluator replay) | §0.1-§0.2; §1.2 R2 note; §1.5 partition; §2.10; §3 M4; §4 C10; test (l); §8 C |
| Bind screenshot hash, dimensions, decode/crop/resize transform, usable-domain digest, signature digest and sampler settings before freeze | §0.2 items 1-5; §2.10/2.13; §3 M4/schema; §4 C11/C14; test (m)(n)(t)(u); §5 Phase A pins; §8 A/C |
| Phase B maps saved pixel indices to separately pinned evaluator ownership projected into exactly that source coordinate space | §0.3; §1.6; §2.11; §3 M5/M6; §4 C12; test (p); §5 Phase B; §8 D |
| Never replay on evaluator RGB or feed back to inference | §0.3; §1.2 R2 note; §1.6 chain; §3 M6; §4 C12; tests (o)(p); §5 leakage guards |
| Typed-unavailable if alignment cannot be proven | §0.3 item 3; §3 M5; §4 C12 + failure rules; test (q) |
| Appearance-proxy limits preserved and extended to source-pixel evidence | §0.4; §6 strict abstention and proxy limits |
| Pre-filter candidates with dispositions + witness/refusal, exact unique match/envelope/end-signature/alternative joins, left-source/right-target orientation, asymmetric tests, retrievable bounded support (R1, retained) | §3 M1-M3; §4 C2-C8; tests (b)-(k); §9.1 |

### 9.3 Luna R2 HOLD closure map (R3)

| R2 HOLD condition | Where made mandatory |
|---|---|
| 1. Name the exact candidate-generation/filter boundary where every admitted match (including later-dropped matches) emits its matcher witness or typed refusal; bind immutable row index and final keep/drop disposition; tests for dropped-candidate witness/refusal and row↔disposition identity | §0.5; §1.5 coverage gap; §2.1/2.12; §3 M1; §4 C13; tests (r)(s); §8 B/D; §9.3 |
| 2. Identify how original screenshot byte SHA/length and `FrameTransform` are captured at initial image loading and passed through the session/signature call; bind byte hash, decoded-pixel digest and transform to the same decoded frame/session; typed refusal if ingress metadata unavailable; ingress-to-signature provenance tests | §0.6; §1.7; §2.13; §3 M4; §4 C14; tests (t)(u)(v); §8 A/D; §9.3/§9.4 |
| Preserve R2 screenshot-sampler witness, Phase B projection-only/no-evaluator-RGB boundary, exact joins, output-neutral default path, bounded payload and abstention contracts | §0.2-§0.4; §3 M2-M3/M5-M6/schema; §4 C1-C12; §5; §6; unchanged |

### 9.4 Lead ingress single-read chain closure map (R4, newest)

| R4 condition (2026-10-04 17:28 UTC Lead clarification) | Where made mandatory |
|---|---|
| Close the full-chain gap: `discover_media` currently hashes `source_path` into `SourceFrame.content_hash` (`input/media.py:60,84-89`) and `load_image` later opens the same path again (`input/image.py:25`); capturing bytes only inside `load_image` would still be two physical reads with a disconnected discovery hash | §0.2 item 1; §0.6 intro + item 1; §1.7 read #1/read #2; §2.13; §3 capture seam (i)/M4; §4 C14; §5 Phase A pins; §6 ingress/digest ordering; §8 A; §9.4 |
| Discovery and decode share one encoded-byte buffer/read (hash + length + de-dup from those bytes, decode from the same bytes), or the redundant discovery read is eliminated and `content_hash` is derived from the ingress record | §0.6 item 1 (variants A shared buffer / B discovery-read removal); §3 M4; §8 A |
| No later path re-hash/re-open (`_sha256(source_path)` or `Image.open(source_path)`) | §0.6 items 1/4; §1.7; §3 M4/schema; §4 C14 + failure rules; §5 Phase A pins; §6; §8 A/final |
| Preserve default output/session behavior (de-dup, stable order, decoded pixels, `session_sha256`; default no-ingress entry byte-for-byte output-neutral) and the typed-refusal boundary | §0.6 item 1/4; §3 test (v); §4 C14/C1; §5; §8 A/final |
| Tests for the single-read chain (one read per screenshot; hash + de-dup + decode from the same buffer; no second read) | §3 test (v) (with (t)(u) extended); §4 C14; §8 D test list |
