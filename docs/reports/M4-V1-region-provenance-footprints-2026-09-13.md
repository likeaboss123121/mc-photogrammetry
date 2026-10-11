# M4 V1: plane-region provenance and sampled footprints

Date: 2026-09-13

V1 carries an optional, image-derived region identity from proposal masks into
`PlaneReading`, then attaches alternative-specific sampled coverage evidence to
`PlaneSignature`. It is an in-memory contract only. The frozen image-runner
serializers, graph score, normal CLI, and reconstruction integration were not
changed.

`PlaneRegionEvidence` binds a stable source ID to the Manhattan axis, sorted
source names, image shape, half-open crop, exact mask digest, and canonical
polygon. The cropped mask is packed with explicit little-endian bit order and
base64 encoding. Decoding validates dimensions, canonical base64, unused
padding bits, digest, source identity, and exact reconstruction. Polygon start
and winding do not affect identity. Proposal masks are copied and intersected
with the caller's usable mask; the source proposals are not mutated.

For each retained distance alternative, `PlaneSignature.region_footprint`
stores its matching `blocks_per_period`, source ID, cell-grid origin and shape,
subsample factor, and integer per-cell counts for in-region samples, valid
samples, and the uniform total sample count. Holes, openings, disconnected
islands, and unusable pixels remain represented by counts. No threshold,
occupancy label, texture-based fill, or polygon/bounding-box fill is produced.
The counts are coverage evidence only. Legacy/manual `PlaneReading` instances
remain valid and report provenance unavailable; their signatures have no
region footprint. Existing reading measurement values, serialized reading
records, signature energy/cell values, and graph matches remain equal in the
synthetic parity controls. Footprints reject a `b` value that is not a retained
alternative on a provenance-bearing reading.

Validation:

```text
.venv\Scripts\python.exe -m pytest -o addopts= tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_region_growth.py -q
62 passed in 49.35s

.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py -k "complete_label_gate or post_accept_buffering" -q
4 passed

.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\reconstruction\lattice_planes.py
passed
```

The synthetic tests cover exact mask round-trip (including packed padding),
L/opening/disconnected shapes, usable-mask exclusion, multiple `b` grids,
proposal reordering and source association, deterministic records, legacy
compatibility, numeric/graph parity, forbidden dataset/evaluation imports, and
invalid-alternative rejection. No R1-R4/E1-E3 artifact was read or modified;
no evaluation or capture was run.

SHA-256 before this report was written:

```text
src/mc_photogrammetry/reconstruction/lattice_planes.py
47e95e4ed5b10a3b5492e0254a49614a5c7d2ada318715a49b37e83e2df58a85
tests/test_lattice_planes.py
132cefaf6c8ecfea510aa92dc87d9e89bc559274cd6a9881afd0dbf632e2c070
```

## Post-acceptance audit corrections

The only production caller of `plane_signature` is the frozen image runner, and
it passes `usable=usable`. Provenance-bearing signatures now reject an omitted
usable mask. A requested `b` must match exactly one retained alternative within
the existing absolute `1e-9` tolerance (with floating-point boundary slack);
the signature and footprint use the alternative's canonical `b` and measured
distance. Zero or multiple matches abstain with `ValueError`.

Region evidence is decoded and validated when constructed and again when
attached to `PlaneReading`, so malformed packed length/base64, digest, source
ID, axis, bbox/crop, pixel count, nonminimal crop, encoding, or bit order cannot
report provenance as available. Tests also prove the per-axis/side cap retains
the corresponding masks and IDs and excludes dropped candidates. A real
rectifier control maps samples back to a rendered image, cuts a hole through
those sampled source pixels, and separately excludes another sampled patch;
the footprint counts decrease as expected.

An image-only rendered-room run through the frozen runner produced proposal
readings and signatures with no signature abstentions. The record key sets and
`schema_version=1` are unchanged; region evidence remains in memory and is not
added to the v1 reading/signature records. The runner hashes `lattice_planes.py`
and binds code hashes into each new artifact ID, so producing a new freeze with
this implementation necessarily yields the new source hash and a correspondingly
different ID for the same inputs. Tied numeric readings now use source-region ID
as the deterministic final ordering key; proposal-reorder tests pin that order.

The immutable R4-L1B freeze loaded under the current strict verifier with its
original bytes unchanged: artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`,
freeze SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`,
and sidecar SHA-256 `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`.
The older R1 freeze was also checked read-only but predates current proposal
configuration keys and fails strict loading at `configuration.proposals.config`;
it was not modified. No R2-R4 or E1-E3 artifact was written or rescored.

Final focused validation:

```text
.venv\Scripts\python.exe -m pytest -o addopts= tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_region_growth.py -q
68 passed in 64.32s

.venv\Scripts\python.exe -m pytest -o addopts= tests\test_frozen_image_runner.py -q
13 passed in 1.24s

.venv\Scripts\python.exe -m pytest -o addopts= tests\test_joint_plane_selector.py -k "budget_exhaustion_discards_all_partial_certificates or post_accept_buffering_budget_abort" -q
3 passed, 55 deselected

.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\reconstruction\lattice_planes.py tests\test_lattice_planes.py tests\test_frozen_image_runner.py tests\test_joint_plane_selector.py
passed
```

Final SHA-256 values:

```text
src/mc_photogrammetry/reconstruction/lattice_planes.py
034f840f676675d9cffe797a34c1fe96409a9acda11b6adf95a6b43c1322d27e
tests/test_lattice_planes.py
a832de2b9429d77a19364d08f0a10923970737d7001b58c251a49a72cad579c4
tests/test_frozen_image_runner.py
e430c3f073f452ec8e640bd4b7aabcc54ee6239ddee1cd4cdf84038aa683c98a
tests/test_joint_plane_selector.py
4c28c16077b66928974093401aaf9ecdd7a868d291e909fe0565de62e4fa1061
```
