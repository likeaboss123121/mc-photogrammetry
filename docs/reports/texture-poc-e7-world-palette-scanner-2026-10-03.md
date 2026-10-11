# E7 read-only bounded Anvil world-state coverage scanner (evaluation-only)

- Task ID: `texture-poc-e7-world-palette-scanner-2026-10-03`
- Task file: `agent swarm instantiation template/tasks/active/texture-poc-e7-world-palette-scanner-2026-10-03.md`
- Worker: DeepSeek (Sol-equivalent), bounded implementation worker.
- Status: implemented; **E7R1 and E7R2 fail-closed repairs applied**, focused synthetic
  checks pass; **stopped for lead review**. No self-acceptance.
- Files written (the only writes): the two owned source/test paths plus this report.
- No real Minecraft save, `.minecraft` directory, launcher, Minecraft install, screenshot,
  JAR or production file was read; no world was scanned. Every fixture is a minimal
  synthetic `.mca`/NBT world built inside a pytest temporary directory.

## 1. Owned paths and hashes

| Path | Role | Bytes | SHA-256 |
|---|---|---:|---|
| `scripts/evaluation/audit_world_region_palette.py` | scanner (CLI, bounded parser, report) | 45,128 | `19092C5524DB8E5716D81008C414FA4D31AE23C367493A64E72896BC94E88DA0` |
| `tests/test_audit_world_region_palette.py` | focused synthetic tests | 44,525 | `0A8A917136CDA881BD23B76D4D495FACF9FA43ECE425CD9776FF3509940543FF` |
| `docs/reports/texture-poc-e7-world-palette-scanner-2026-10-03.md` | this report (not self-hashed; editing it would change its digest) | — | — |

The scanner's `ENGINE_VERSION` is now `e7-1r2`, recording the E7R1 and E7R2 repair
revisions in every emitted payload. No file outside the three owned paths was edited;
pre-existing dirty hunks elsewhere in the tree were not touched.

## 2. What the scanner does

`audit_world_region_palette.py` is developer/evaluation scaffolding only. It answers one
question: which exact BlockStates exist inside explicitly named, bounded block regions of
explicitly named, already-present saves. It never runs Minecraft, never writes to a save,
and is not an input to the reconstruction program.

CLI:

```
python scripts/evaluation/audit_world_region_palette.py \
    [--game-dir PATH] --regions-json PATH
```

- `--game-dir` defaults to the platform's usual `.minecraft` (`%APPDATA%/.minecraft`, else
  `~/.minecraft`). The directory is only read, never created.
- `--regions-json` is required and must be a strict JSON object:

```json
{
  "schema": "mc-photogrammetry-world-region-scan-request-v1",
  "regions": [
    {"save": "Alpha", "min": [0, 0, 0], "max": [64, 64, 64], "label": "optional"}
  ]
}
```

- Top-level keys are limited to `schema` and `regions`; each region is limited to
  `save`, `min`, `max` and (optional) `label`. `min` is inclusive, `max` exclusive.
- Save names must be single path components and must resolve directly under
  `<game-dir>/saves`. Traversal, separators, drive/absolute names, trailing dots/spaces,
  NUL bytes and symlink/junction save, region-directory or `.mca` paths are refusals.
- Each region's per-axis edge must be `<= MAX_REGION_EDGE = 512` and its volume
  `<= MAX_REGION_VOLUME = 250_000`, imported from
  `mc_photogrammetry.dataset.schemas` so the scanner cannot accept a region the capture
  factory would later reject. At most 64 regions are accepted per request.

Only the `.mca` files and only the 32x32 chunk slots intersecting the requested regions
are opened. No save is enumerated, no other save is opened, no unselected chunk payload is
read, and nothing is staged or copied. Region files are opened read-only; each requested
chunk is read by sector seek *after* the whole header has been validated.

Output: one strict canonical JSON document on stdout (sorted keys, no whitespace, `NaN`
free), nothing else on stdout. It contains per-region exact non-air state counts
(`Name` plus sorted `Properties`, rendered as `name[key=value,...]`), non-air and air
totals, scanned/requested/missing chunks, requested/present/missing sections, observed
`DataVersion` values, pairwise exact-state intersections with coverage ratios and Jaccard,
documented limits/supported formats, and a deterministic SHA-256 seal over the canonical
document (the `seal` field itself excluded). Refusals emit a single sealed JSON refusal
document on stdout and exit `2`; successes exit `0`.

## 3. Documented format support

Supported and decoded:

| Item | Supported set |
|---|---|
| Region container | `.mca` only, `<save>/region/r.<x>.<z>.mca`, overworld |
| Header | two sectors: 1024 big-endian location entries (3-byte offset, 1-byte sector count) plus timestamp sector |
| Chunk frame | 4-byte big-endian declared length (`compressed + 1`), 1 compression byte, payload |
| Compression | `1` gzip, `2` deflate (zlib), `3` none |
| Chunk layout | post-1.18 root-level `sections` list with per-section `block_states` |
| Palette | `Name` (namespaced; an un-namespaced name is normalized to `minecraft:`) plus optional `Properties` of string→string; property keys sorted |
| Index packing | modern non-spanning packing, `bits = max(4, (palette_size - 1).bit_length())`, fixed 4096 entries; singleton palette with no `data` accepted |
| Chunk state | `DataVersion` in `[2865, 6000]` (1.18 release … deliberate upper guard), `Status = "minecraft:full"`, `xPos`/`zPos` when present must match the slot |
| NBT strings | strict Java modified UTF-8: one/2/3-byte forms, `C0 80` NUL (raw `00` also accepted, as `readUTF` does), CESU-8 surrogate pairs combined to the supplementary character |
| State keys | delimiter-safe: `\`, `[`, `]`, `,`, `=` inside a name/property key/property value are backslash-escaped |
| Coordinates | negative chunk/region coordinates and region files on negative indices |

Refused, fail closed (never a warning):

- a game directory, `saves` root, save directory, region directory or `.mca` file that is
  link-like (symlink or Windows reparse point), **any link-like ancestor** of those paths,
  a resolved `saves` root whose resolved parent is not the resolved game directory, and a
  resolved save that escapes the resolved `saves` root;
- external `.mcc` chunks (compression byte bit `0x80`), reported with the expected
  `c.<chunkX>.<chunkZ>.mcc` companion name;
- LZ4 (`4`), ZSTD/SectorFile and every compression id outside `1/2/3`;
- pre-1.18 `Level` chunk layout; `DataVersion` outside the range; non-`minecraft:full`
  `Status`; chunk `xPos`/`zPos` mismatching the region slot;
- **half-zero location entries** (exactly one of offset/sector count set), an occupied
  allocation whose full `sector_count * 4096` span runs past the region file end, and any
  overlap between two occupied allocations. This is validated for **all 1024 header slots**
  from the two header sectors alone, so an overlap involving an unselected chunk slot
  refuses the scan without that payload ever being read;
- palette entries with keys other than `Name`/`Properties`, non-string properties,
  `data` that is not a `LongArray`, a data array of the wrong long count, and any palette
  index `>= palette_size`;
- NBT strings that are not strict Java modified UTF-8: a sequence truncated by the string
  end, a malformed continuation byte, an overlong form other than `C0 80`, a raw four-byte
  UTF-8 sequence (modified UTF-8 has no four-byte form), or an unpaired UTF-16 surrogate;
- malformed sector locations (offset into the header), frames that overflow their allocated
  sectors, truncated/garbled compressed streams, trailing bytes after the compressed stream
  or after the NBT root, truncated NBT, NBT beyond the depth/element budget, duplicate
  compound keys, and a missing required region file.

Explicit byte bounds: compressed chunk payload `<= 255 * 4096 = 1,044,480` bytes (one
internal Anvil allocation); decompressed chunk `<= 64 MiB`, enforced through
`zlib.decompressobj(..., max_length)` without unbounded allocation; NBT depth `<= 24`;
NBT element budget `2,000,000`. `decompressobj` also rejects trailing bytes and
non-terminated streams.

Air-family states (`minecraft:air`, `minecraft:cave_air`, `minecraft:void_air`, matching
`mc_photogrammetry.dataset.blocks.AIR_BLOCKS`) are counted but aggregated into `air_total`;
`state_counts` holds only exact non-air BlockStates, so `non_air_total == sum(state_counts)`.

## 4. Fail-closed repairs (E7R1 numbered list, then E7R2)

1. **Link containment for game dir / saves root / intermediates.** `_assert_link_free`
   walks the leaf and every ancestor with `is_link_like` (symlink or Windows reparse
   point) and refuses the first link-like component. It now guards the `--game-dir` root,
   `<game-dir>/saves`, the save directory, the `region` directory and the `.mca` file. The
   resolved `saves` root must have the resolved game directory as its direct parent, and a
   save must still resolve inside that root. A linked intermediate directory can no longer
   redirect the read outside the bounded root.
2. **Full sector-allocation EOF bounds.** `_location_table` validates every occupied slot's
   entire `(offset + sector_count) * 4096` span against the file size, not just the declared
   payload. A chunk whose payload would fit but whose claimed allocation runs past EOF is
   refused.
3. **Half-zero location entries.** A slot with exactly one of offset/sector count set is
   malformed and refuses the whole scan, in any of the 1024 slots.
4. **Overlap with any occupied header slot.** All occupied allocations are sorted and any
   pairwise overlap refuses the region file. The check uses only the header sectors, so an
   overlap that involves an unselected chunk slot still refuses the scan and that chunk's
   payload is never read.
5. **Java modified UTF-8 NBT strings.** `_decode_modified_utf8` decodes NUL (`C0 80`, plus
   raw `00` for `readUTF` parity) and CESU-8 surrogate pairs, and rejects truncated or
   malformed sequences, overlong forms other than the NUL form, four-byte UTF-8, and
   unpaired surrogates instead of silently replacing them.
6. **Delimiter-safe exact state keys.** `_escape_component` backslash-escapes `\`, `[`,
   `]`, `,`, `=` inside each name/property-key/property-value; the renderer's own `[`, `,`,
   `]`, `=` stay structural. Two genuinely different states can no longer collapse to one
   `state_counts` key, and ordinary states such as `minecraft:oak_log[axis=x]` render
   exactly as before.
7. **NBT depth/element and compressed-bound tests.** Dedicated synthetic tests now cover the
   depth bound, the element budget, the array-length element check, and both compressed and
   decompressed chunk byte bounds.
8. **DataVersion 2865 `Status` compatibility verified.** The Minecraft Wiki chunk-format
   page documents the 1.18 `Status` values as namespaced (one of `minecraft:empty` …
   `minecraft:full`, with `minecraft:full` the only fully generated state). A boundary test
   now scans a `DataVersion = 2865` (`MIN_DATA_VERSION`) chunk with
   `Status = "minecraft:full"` and asserts it is accepted.
9. **Payload-fits-but-allocation-past-EOF test.** A dedicated test declares a payload that
   fits its one real sector while the header claims three sectors, and asserts the
   allocation-EOF refusal.

### E7R2 fail-closed repairs (relative-path link guard + exact byte boundaries)

- **Relative paths are now guarded against their cwd ancestors.** `_assert_link_free`
  previously walked `(path, *path.parents)` directly. For a relative path such as `game` or
  `.` the `parents` sequence stops at `.`, so every directory above the current working
  directory escaped the link guard. The argument is now converted with `os.path.abspath` -
  lexical only: it joins the cwd and collapses `.`/`..` but never resolves a symlink - and
  the leaf and *all* parents of that absolute form are checked. The absolute form is
  returned, so the game directory, `saves` root, save, `region` directory and `.mca` file
  callers all keep guarding the same validated path. A relative or dot `--game-dir` whose
  cwd ancestor is link-like is now refused instead of silently followed.
- **Exact byte boundaries locked in.** Focused tests pin both caps as `<=`: a gzip (`1`)
  and a zlib (`2`) chunk decompressing to exactly `MAX_DECOMPRESSED_CHUNK_BYTES` is
  accepted and one decompressed byte over is refused; a frame whose compressed payload is
  exactly `MAX_COMPRESSED_CHUNK_BYTES` is accepted and one byte over is refused (the
  pre-existing over-bound test is retained).

## 5. Exact checks performed

All commands were foreground, project-venv (`.venv\Scripts\python.exe`, Python 3.10.11):

| Check | Command | Result |
|---|---|---|
| Byte-compile | `python -m py_compile scripts/evaluation/audit_world_region_palette.py tests/test_audit_world_region_palette.py` | exit `0` |
| Focused synthetic suite | `python -m pytest -o addopts='' -q --no-header tests/test_audit_world_region_palette.py` | **93 passed** (observed 1.48s after the E7R2 additions; 83 before) |
| E7R2 link guard | tests `test_link_like_cwd_ancestor_is_refused_for_a_relative_game_dir[game]` and `[.]` | relative and dot `--game-dir` with a mocked link-like cwd ancestor both refuse with a `link-like` message |
| E7R2 exact byte bounds | tests `test_decompressed_chunk_byte_bound_accepts_exactly_the_limit` / `..._refuses_one_byte_over` (gzip `1`, zlib `2`) and `test_compressed_chunk_byte_bound_accepts_exactly_the_limit` / `..._refuses_one_byte_over` | decompressed exactly at the cap accepted, one byte over refused; compressed payload exactly at the cap accepted, one byte over refused |
| Scoped whitespace check | `git diff --check -- scripts/evaluation/audit_world_region_palette.py tests/test_audit_world_region_palette.py` | exit `0` (both files are untracked, so `git diff` sees no hunks) |
| CLI stdout/write check | test `test_cli_subprocess_emits_json_only_on_stdout_and_writes_nothing` | real subprocess: exit `0`, stderr empty, exactly one newline on stdout, stdout parses as one JSON document, game tree byte-identical before/after |
| No-write check on success and refusal | test `test_success_and_refusal_write_nothing_under_the_game_directory` | game tree (sizes + SHA-256) identical after both a success run and a refusal run |
| Determinism and seal | test `test_stdout_is_one_deterministic_sealed_json_document` | two runs byte-identical; `seal == digest_json(document without seal)` |

Note: `ruff` is not installed in the project venv, so no ruff pass was run; the byte-compile
and pytest runs above are the executed checks.

Synthetic coverage in the focused suite (fixtures written by the test file's own
region/NBT encoder, not the scanner):

- exact block counts and canonicalization: sorted property keys, no air in `state_counts`,
  `non_air_total == sum(state_counts)`, delimiter characters in names/properties that would
  otherwise collide stay distinct;
- singleton palette with no `data` array, packed palettes below and above four bits
  (4-bit `bits`, 5-bit `bits`), partial y windows, empty sections without `block_states`;
- boundary coordinates: exclusive `max` at a chunk edge, a region spanning two `r.*.mca`
  files (chunk 31/32 across the 512-block boundary), and negative chunk/region coordinates
  (`r.-1.-1.mca`);
- missing chunks counted, missing section spans counted, missing region file/directory
  refused;
- gzip/deflate/uncompressed decode; external `.mcc`, LZ4, over-bound compressed payload,
  over-bound decompressed payload refused; NBT depth bound, element budget and array-length
  element check refused; **E7R2 exact boundaries: gzip and zlib decompressing to exactly
  `MAX_DECOMPRESSED_CHUNK_BYTES` accepted and one byte over refused, and a compressed
  payload exactly `MAX_COMPRESSED_CHUNK_BYTES` accepted with one byte over refused**;
- corrupt sector offset, offset into the header, **half-zero location entries (both
  orientations, in an unselected slot)**, **allocation past EOF with a fitting payload**,
  **overlap with an unselected occupied slot**, truncated region file, trailing bytes after
  the NBT root, wrong-length data array, non-`LongArray` data, invalid palette index,
  multi-entry palette without `data`;
- Java modified UTF-8: `C0 80` NUL name and CESU-8 surrogate-pair name decode exactly;
  truncated two-byte sequence, malformed continuation, overlong `C1 81`, raw four-byte
  UTF-8, unpaired high surrogate and unpaired low surrogate all refuse;
- pre-1.18 `Level` layout, out-of-range `DataVersion`, `DataVersion = 2865` with
  `minecraft:full` accepted, non-full `Status`, `xPos` mismatch;
- request safety: unsafe save names (traversal, separators, absolute/drive, trailing dot,
  leading space, NUL, empty), link-like game directory / `saves` root / intermediate
  directory / save directory / region file guards, **E7R2: relative `game` and dot `.`
  `--game-dir` with a mocked link-like cwd ancestor both refused**, unknown request/region
  keys, boolean coordinate, wrong request schema, empty region list, edge cap, volume cap,
  inverted region, unknown save, missing `--regions-json`;
- pairwise intersection/coverage/Jaccard on known counts and `null` coverage for a
  zero-non-air region.

## 6. Limitations and honest non-claims

1. **Overworld only.** Only `<save>/region/*.mca` is read. Nether/end (`DIM-1`, `DIM1`)
   and entity/poi region files, and the newer SectorFile (`.sf`/`.sfe`) container, are not
   supported and are not opened.
2. **No `.mcc`/LZ4/ZSTD.** External chunks and non-gzip/deflate/uncompressed frames make the
   whole scan refuse rather than partially decode.
3. **DataVersion guard.** Versions above `6000` are refused even if a future layout stayed
   compatible; the supported layout claim is exactly the post-1.18 root-`sections` format.
   `DataVersion = 2865` with `Status = "minecraft:full"` is verified accepted (section 4.8).
4. **Full chunks only.** A single non-`minecraft:full` chunk refuses the scan; proto-chunks
   are never counted as final states.
5. **Strict Java modified UTF-8 only.** Standard four-byte UTF-8 sequences are refused as
   malformed for the modified encoding. A modern world that writes standard UTF-8 non-BMP
   characters into any chunk NBT string would therefore refuse the scan rather than decode
   it; this is deliberate fail-closed behavior, not a silent conversion.
6. **Whole-header allocation validation.** Any half-zero entry, out-of-file allocation or
   overlap in *any* of the 1024 slots refuses the entire region file, even when the affected
   chunk would not have been selected or counted. This is intentional: the scanner never
   reads an unselected payload, so it also never trusts an inconsistent allocation map.
7. **Ancestor reparse points refuse, relative paths included.** The guard walks the leaf
   and every parent of the *lexically-absolute* path, so a game/saves path reached through
   a redirected ancestor - a junctioned `Documents`/`AppData` location, or any link-like
   directory above the cwd of a relative `--game-dir` - is refused rather than followed,
   because the containment claim would otherwise be false. The absolutization is lexical
   (`os.path.abspath`) and never resolves a symlink, so the check itself cannot be
   redirected.
8. **Modded palette keys refuse.** Palette entries carrying extra keys (e.g. mod data) are
   refused instead of silently narrowed to `Name`/`Properties`.
9. **Sections outside the requested y window are not decoded.** This is deliberate bounded
   work; a malformed palette in an unreferenced section is never inspected.
10. **Air-family states are aggregated.** Exact counts are exact for non-air states; air is
    reported only as a total.
11. **No suitability verdict.** Pairwise overlap/coverage is a descriptive exact-count
    statistic. Palette counts alone do not prove a world is suitable, and no threshold or
    pass/fail bar is defined here.
12. **Seal semantics.** The seal proves the emitted JSON is deterministic and internally
    consistent; it does not prove the source save was unchanged during the read.
13. **No locking.** A save being written concurrently could be read inconsistently; this is
    an offline developer refinement tool.
14. **`level.dat` is not read,** so the scan cannot confirm a directory is a world or report
    its `LevelName`; it reads only the named save's region files.
15. **Bounded by construction, not by runtime measurement.** No live memory/time profile was
    taken; the bounds are the documented caps above.

## 7. Prohibitions honored

No real user-world scan, no launcher/Minecraft activity, no screenshot/image read, no
staging/copying, no dependency install/download, no background process/job/workflow/
subagent, no junction/reparse-point creation or operation, no recursive deletion, no
encoded/obfuscated command, no generated executable, no persistence or Defender/system
change, no system directory write, no commit, and no source edit outside the three owned
paths. The scanner's own tests create and delete only pytest temporary directories.

## 8. Stop condition

Implementation, E7R1 and E7R2 repairs, synthetic tests and this report are complete.
**Stopped for lead review.** Do not launch any real-world scan or treat any palette count as
a suitability or reconstruction result until the lead accepts this task.
