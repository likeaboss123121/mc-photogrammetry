# M4 P1-A Periodic-Energy and Match-Shift Contract Audit

Date: 2026-09-13
Role: DeepSeek read-only/no-shell pre-implementation audit
Status: directive refinement; no implementation or evaluation performed

## Required semantics

The new evidence records the phase of periodic structural edge energy. It does not identify a block
seam, block residue, lateral origin, or world coordinate.

For each nonnegative 1-D profile and period:

- `response = abs(sum(profile * exp(-2*pi*i*offset/period)))`.
- `mass = sum(profile)`.
- `concentration = response/mass`, bounded to `[0,1]` up to floating error.
- availability is `measured` only for finite input, period at least 2, mass above the existing `1e-9`
  degeneracy guard, and response above the existing `1e-12` guard.

No concentration threshold decides usefulness. A measured phase of exactly zero remains `measured`;
the explicit status distinguishes it from the legacy zero sentinel.

`phase_pixels=(u,v)`: u is the column profile (`edge_patch.sum(axis=0)`), v is the row profile
(`sum(axis=1)`). Evidence binds the exact reading period, rectification centre, extent, samples and
optional source-region ID. Legacy `phase_pixels` agrees exactly with measured evidence and stays zero
on unavailable axes. Evidence must not enter `reading_order_key`.

## Minimal additive types

An optional frozen/slotted `PeriodicEnergyPhaseEvidence` belongs on `PlaneReading`. It contains only
plain tuples, floats, strings and optional source ID: period, patch centre/extent/samples, phase `(u,v)`,
per-axis status, response magnitude, profile mass and concentration. Manual/legacy readings default
to `None`. Construction validates finite/range/status consistency and binds evidence fields to the
containing reading.

An optional frozen `PlaneMatch.cell_shift_rc=(row_shift,column_shift)` retains the admitted
correlation peak. Manual/legacy matches default to `None`.

## Exact shift convention

The current FFT order is `FFT(first)*conj(FFT(second))`, with `first=warped_left` and
`second=right.cells`. Therefore the retained peak satisfies:

`warped_left[x + cell_shift_rc] ~= right[x]`.

For the existing `np.roll(base,(3,-4))` control, the exact expected shift is `(-3,+4)`. The current
test permits both signs and must be replaced with asymmetric non-square, positive/negative row and
column controls. Shift order is `(row/v,column/u)`. It is residual after both homographies and
`origin_uv` translations; it has no absolute/world meaning.

## Integration hazards

1. `match_plane_graph` rebuilds a `PlaneMatch` field-by-field; it must copy the new shift or graph
   output silently loses it.
2. `frozen_image_runner._match_record` currently serializes `asdict(match)`. The additive field would
   violate strict `_MATCH_KEYS`. The record writer must explicitly retain the legacy key set; neither
   `_MATCH_KEYS` nor `_READING_KEYS` changes.
3. Old artifact/scripts instantiate `PlaneMatch(**record)` without the field, requiring default `None`.
4. Selector keys, scores and admission must ignore the shift. Additive evidence changes dataclass
   equality/hash, so parity tests should compare existing semantic fields or evidence-stripped copies.
5. A non-finite profile currently can yield a non-finite legacy phase. The structured path must mark it
   unavailable and normalize the legacy sentinel to zero; document this behavior correction.
6. New source hashes change newly generated artifact IDs, while immutable old artifacts must still
   strict-load from their stored content.

## Decisive tests

- Exact signed shifts for asymmetric non-square arrays and each signed row/column direction.
- Shift propagation through `match_plane_graph` and `annotate_integer_residuals`.
- u/column versus v/row evidence, including one measured and one unavailable axis.
- Genuine zero phase versus empty, non-finite and zero-resultant profiles.
- Pins around mass `1e-9` and response `1e-12`; legacy/evidence agreement.
- Concentration equality, range and deterministic repeatability.
- Malformed evidence: lengths, status, NaN/inf, negative mass, concentration, phase range, reading
  geometry/source mismatch.
- Evidence-present versus stripped graph/selector semantic parity.
- Frozen schema-v1 key stability and immutable-artifact loading.
- Import/signature isolation from dataset, evaluator, manifest, reference, ground truth, supplied
  camera and manual metadata.
