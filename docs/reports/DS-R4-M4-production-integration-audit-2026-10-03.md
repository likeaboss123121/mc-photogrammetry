# DS-R4 read-only M4 production integration audit (2026-10-03)

Worker: DeepSeek (`deepseek-flash`), read-only/no-shell patch.
Status: READ-ONLY AUDIT COMPLETE. No existing file was edited, no test/image/workload
was run, no manifest or capture was altered. Stop for lead review and Jev routing.

## Delivery note (tooling blocker on the comms append)

The task asked for this report in `AGENT_COMMS.md`. That file is **not valid UTF-8**
(the read tool refuses it, and the grep tool reports invalid-UTF-8 lines around
2280, 3242 and the tail). The file-observation policy therefore blocks `edit`
("file has not been read"), and `write` would replace the whole file, destroying
content. To preserve every existing artifact, the audit is recorded here as a
linked report. The lead (or a direct-shell worker) should append the entry in
section 7 to `AGENT_COMMS.md`; it is ASCII-only and ready to paste.

## 1. Trace: `image-derived-output` -> `.litematic` (exact source, APIs, call sites)

1. CLI parser/dispatch: `src/mc_photogrammetry/cli.py` - `image-derived-output`
   (parser L191-208, dispatch L387-418), `image-derived-deliver` (parser L210-231,
   dispatch L420-448), generic `export` (L736-749).
2. Production facade: `src/mc_photogrammetry/pipeline/image_derived_production.py`
   - `derive_image_output_from_paths` (L134), `ImageDerivedProductionResult` (L57).
   It returns the staged result; it saves nothing itself.
3. Camera/session + connections + pose graph + first persistence:
   `src/mc_photogrammetry/pipeline/image_derived_entry.py` -
   `build_image_derived_entry` (L388): `discover_media`/`load_image` ->
   `image_derived_session.build_image_derived_session` ->
   `image_derived_connections.build_image_derived_connections` (production matcher
   `joint_matching.match_and_solve_pair`) ->
   `image_derived_pose_graph.compose_image_derived_pose_graph` ->
   `_persist_session` (L619), which writes a schema-2 `ScanProject` carrying only
   `image_derived_session` and the default empty `BlockVolume`.
4. Lattice/plane measurement:
   `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` -
   `measure_image_derived_lattice` (L819): `_reconstruct_calibration` from retained
   pixels -> `region_growth.propose_image_planes_v2` ->
   `lattice_planes.frame_plane_readings_from_proposals` -> structural-weight/P0
   split provenance -> `plane_geometry_provenance.bind_geometry_provenance`.
5. Geometry orchestration:
   `src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py` -
   `derive_image_derived_geometry` (L208), stage order `_STAGES` (L33) / `_READY`
   (L34); the first non-ready stage returns typed `unavailable`. Fixed gates at
   L249-253: `max_shift_cells=32, integer_tolerance=0.15, min_resultant=0.9,
   ambiguous_margin=0.02`. Stage `g` = `image_derived_graph.derive_image_derived_graph`
   (`image_derived_graph.py` public L1337, private L1233), which calls
   `plane_graph.match_plane_graph` (L1291) then `plane_graph.solve_component` (L1297)
   and returns `_unavailable` on `solution.reason` (L1303-1322).
6. Output / state fusion:
   `src/mc_photogrammetry/pipeline/image_derived_output.py` -
   `derive_image_derived_output` (L296) -> `_derive_ready` (L232):
   `plane_placement.resolve_face_placement` -> `plane_volume.build_plane_volume` ->
   `plane_face_observation.frame_context_from_readings` /
   `build_plane_face_observations` -> `appearance_state_proposals.propose_face_states`
   -> `state_promotion.promote_plane_states` (`state_promotion.py` L994). Frozen
   policy: `image_derived_output_policy.py` `IMAGE_DERIVED_OUTPUT_POLICY` (L191);
   `allow_omitted_unknown=False` (L185); volume `max_omitted_unknown=1` (L173).
7. Delivery: `src/mc_photogrammetry/pipeline/image_derived_output_delivery.py` -
   `deliver_image_derived_output` (L55) requires `status=="promoted"`,
   `export_ready`, `exact_state_eligible`, `omitted_unknown==0` (L41-49), then
   delegates once to `image_derived_delivery.deliver_image_derived_litematic`
   (`bounds=None`, one `LitematicExporter.export`). Wrapper:
   `image_derived_production_delivery.py` - `derive_and_deliver_image_output` (L114).
8. Exporter: `src/mc_photogrammetry/export/litematic.py` -
   `LitematicExporter.export` (L46); L59-60 raise
   `LitematicExportError("cannot export an empty volume")`.
9. Persistence: `src/mc_photogrammetry/domain/models.py` - `ScanProject` (L163)
   carries only `volume` + `image_derived_session` (`_SCHEMA2_PROJECT_KEYS` L153);
   `src/mc_photogrammetry/project/store.py` - `save_project` (L22) /
   `load_project` (L45) validate/canonicalize the session record only.

## 2. Verified integration gaps (source-backed)

- **G1 - the image-derived `.mcscan` never carries reconstruction geometry.**
  `image_derived_entry._persist_session` (L619) builds
  `ScanProject(name=..., source_paths=..., schema_version=2,
  image_derived_session=...)` with the default empty `BlockVolume`; no
  image-derived module assigns `project.volume` or persists the R1/R2-A records.
  Consequence: `image-derived-output --project p.mcscan` followed by
  `mc-photogrammetry export p.mcscan out.litematic` reaches
  `LitematicExporter.export` with an empty volume and raises
  `LitematicExportError("cannot export an empty volume")` (litematic.py L59-60).
  The documented `--project` audit path cannot reach the exporter.
- **G2 - delivery structurally requires a zero-omission volume.**
  `exact_state_eligible == (omitted_unknown == 0)` (`state_promotion.py`
  L964/L988) with `allow_omitted_unknown=False` (`image_derived_output_policy.py`
  L185) means any unknown cell yields a typed `review` (`promotion_omitted_unknown`),
  no blocks, and a delivery refusal (`image_derived_output_delivery.py` L41-49).
  The one accepted genuine recipe keeps 66 contributions over 58 occupied
  coordinates plus 15 unknown omissions
  (`docs/reports/M4-V3-supported-volume-bridge-2026-09-19.md` section 2/3), so its
  composed promotion is ineligible and the M6 facade cannot reach its
  successful-delivery branch (`docs/reports/M6-image-derived-orchestration-2026-09-19.md`,
  "Known coverage boundary", L255-266).
- **G3 - real-image geometry never assembles.** `derive_image_derived_geometry`
  stops at stage `g` on every recorded normal/frozen run: the 2026-10-03 v3b
  attempt returns `plane_integer_residual` with first failed predicate
  `minimum_resultant` and resultants `.39146/.24717/.24521` against `.90`; the
  L1B/R4 eight-view Canal Town freeze returns `ambiguous_quadrant`. R2-A therefore
  returns `unavailable` (`geometry_unavailable`) and nothing reaches promotion.

## 3. Accepted structural seams vs. verified rendered-fixture behavior

- Accepted structural seams (bounded, replay-validated code plus focused tests and
  recorded digests): M7 entry, M9-C lattice/provenance, R1 geometry orchestration,
  R2-A/P output + policy, R2-D delivery, M4-V1/V2/V3, M5 plane-face observation /
  appearance proposals / state promotion, schema-2 session persistence, M6
  record-driven orchestration facade. These are code acceptances, not
  rendered-fixture proof.
- Verified rendered-fixture behavior: the furthest genuine image-only run reaches
  is a graph-stage abstention (`plane_integer_residual` on the four frozen PNGs;
  `ambiguous_quadrant` on the 8-view R4 freeze). No accepted rendered fixture
  reaches geometry assembly, appearance fusion, promotion, or `.litematic` export.
- Does a current end-to-end fixture reach export/promotion? **No.** Corroborated by
  `AGENT_HANDOFF.md` L3/L6/L81 ("a genuine rendered fixture still has not reached
  exact-state promotion"), `agent swarm instantiation
  template/project/STATUS.md` L19/L86, and the v3b/R4 results above.

## 4. Recommended single next tranche - M4-V4b: persist and export the assembled image-derived production result

Rationale: this is the only remaining item of the accepted V4 contract
(`docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md` section V4:
"one multi-image image-derived entry ... and review output", "a reviewed schema-2
`.mcscan` envelope with evidence blobs, digests, alternatives, gauge/export
transforms, review state"). It closes a pure production-integration gap (G1), is
bounded, consumes only accepted seams, changes no gate/threshold/radius/matcher,
and makes an assembled result survive into the already-working
`export`/`LitematicExporter`. It is deliberately independent of the
measurement-gated solver evidence (G3), which no implementation may change before
the separately authorized v4 measurement.

Proposed contract (for Astra):

- New canonical, digest-bound `image_derived_production` record emitted by
  `image_derived_production.py` through `to_record`/`from_record` (schema +
  producer + version, stage/status/reason/detail, entry/lattice/geometry/output
  digests, volume/promotion digests, and the typed review facts:
  `omitted_unknown`, conflicts, gauge, `normalization_offset`,
  `scale_state`/`normalization_state`, `export_bounds=None`). No pixels, no
  manual metadata.
- Persist it as a new optional top-level `ScanProject` key through
  `domain/models.py` and `project/store.py`, validated exactly like
  `image_derived_session`; preserve schema-1 and schema-2 loading. Astra decision
  required: bump to schema 3 or extend the schema-2 allowed key set, because the
  current schema-2 reader rejects unknown top-level keys by design
  (`models.py` L228-230).
- On a `promoted`/`exact_state_eligible` output, persist
  `output.promotion.to_block_volume()` into `project.volume` so the existing
  `export` writes the exact-state `.litematic` without rerunning. On a
  non-promotable assembled result, persist the typed record and (only if the lead
  wants a review artifact now) the accepted
  `plane_volume.to_geometry_only_preview(...)` volume, which is `kind=INFERRED`,
  zero confidence and `exact_state_eligible=False`; omitted cells are never given
  a block. Otherwise persist no volume.
- Call sites: persist from `derive_image_output_from_paths`
  (`image_derived_production.py` L134) after the output stage;
  `image-derived-output`/`-deliver` signatures and caps unchanged; `export`
  (`cli.py` L736) unchanged.
- Invariant: normal mode still consumes only ordinary images plus pipeline-owned
  evidence; no manifest/F3/coordinate/direction/bound/reference/scene input is
  added to any public signature; `--project` stays an audit path; no gate, radius,
  tolerance, matcher, solver, joint-selector, capture or launcher change.

Focused acceptance controls:

1. Round-trip: the accepted genuine recipe (reuse `tests/test_plane_placement.py`
   via `tests/test_plane_volume.py` / `tests/test_image_derived_orchestration.py`)
   persisted -> loaded gives an equal envelope digest, summary facts and volume;
   save -> load -> save is byte-identical; schema-1 and schema-2 files still load
   unchanged.
2. Promoted export: the persisted promotion's `to_block_volume()` exported by the
   existing `export` command yields the same block count/coordinates/states as
   `LitematicExporter` over that volume.
3. Review/refusal atomicity: a `review`/`unavailable`/`refusal` result persists a
   typed record with zero fabricated blocks; a persisted geometry-only preview
   contains exactly the V3 preview blocks (`INFERRED`, zero confidence,
   `exact_state_eligible=False`) with omitted cells absent; `export` on a
   volume-less review project refuses as a typed error rather than raising an
   uncaught `LitematicExportError`.
4. Hostile input: stale or resealed envelope digest, swapped
   session/lattice/geometry/output binding, forbidden top-level metadata keys, and
   non-finite/overflowing numeric facts are typed refusals with no partial write.
5. Isolation/no-pixels: recursive key scan proves no `image`/`pixels` payload in
   the envelope; the current production API signature and forbidden-token scan
   (`tests/test_image_derived_production.py` L94-105) still pass; mutating a
   returned `BlockVolume`/state cannot change the record or its digest.
6. Determinism: two runs over the same accepted records give identical envelope
   bytes/digests.
7. Scope proof: no matcher/solver/joint-selector, gate, capture, CLI-cap or
   advanced-mode change; no manual-metadata parameter is introduced.

## 5. Alternatives considered and not selected

- Solver/matcher/alias production repair (G3): not implementable on current
  evidence; the lead/Astra have required a separately authorized v4 measurement
  first and the fresh v4 run is pending exact user authorization (2026-10-03
  entry). Implementing a repair now would be an unjustified algorithm/gate change.
- Relax `allow_omitted_unknown` / `min_resultant` / `integer_tolerance`:
  explicitly forbidden; it would promote incomplete geometry.
- Decouple the entry gate (G2) so lattice/geometry runs when the SIFT
  connection/pose graph is refused: `_connection_constraints`
  (`image_derived_graph.py` L1164) refuses on a non-`ok` graph, and passing `None`
  drops the solved yaw-quadrant constraints the solver consumes. No evidence it
  helps, and it weakens provenance.
- Wire the accepted M6 facade into the CLI: R2-A already composes the same
  accepted records live; M6 is record-driven and referenced only by tests. A
  structural refactor with no output change.
- Surface-local support
  (`docs/reports/DS-R3-AR-surface-local-support-contract-2026-09-20.md`):
  explicitly evaluation-only ("No production matcher, thresholds, camera plans,
  launcher, or metadata path changes") and gated behind the targeted-capture
  preflight, so it is not a production tranche.

## 6. Exact evidence inspected (read-only; hashes below are recorded bindings, not recomputed here)

Source: `cli.py`, `pipeline/image_derived_production.py`,
`pipeline/image_derived_production_delivery.py`, `pipeline/image_derived_entry.py`,
`pipeline/image_derived_lattice_measure.py`,
`pipeline/image_derived_geometry_orchestration.py`, `pipeline/image_derived_graph.py`,
`pipeline/image_derived_output.py`, `pipeline/image_derived_output_policy.py`,
`pipeline/image_derived_output_delivery.py`, `pipeline/image_derived_delivery.py`,
`reconstruction/plane_volume.py`, `reconstruction/state_promotion.py`,
`domain/models.py`, `project/store.py`, `export/litematic.py`.

Tests: `tests/test_image_derived_production.py`,
`tests/test_image_derived_production_cli.py`,
`tests/test_image_derived_production_delivery.py`,
`tests/test_image_derived_output.py`, `tests/test_image_derived_output_delivery.py`,
`tests/test_image_derived_delivery.py`,
`tests/test_image_derived_geometry_orchestration.py`,
`tests/test_image_derived_lattice_measure.py`,
`tests/test_image_derived_orchestration.py`, `tests/test_plane_volume.py`,
`tests/test_plane_placement.py`, `tests/test_state_promotion.py`,
`tests/test_project_store.py`, `tests/test_litematic.py`.

Reports/handoff: `docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md`,
`docs/reports/M4-V3-supported-volume-bridge-2026-09-19.md`,
`docs/reports/M4-schema2-image-session-persistence-2026-09-19.md`,
`docs/reports/M6-image-derived-orchestration-2026-09-19.md`,
`docs/reports/M5-state-promotion-2026-09-19.md`,
`docs/reports/M5-image-derived-delivery-2026-09-19.md`,
`docs/reports/M4-V2-fixture-reachability-2026-09-19.md`,
`docs/reports/LEAD-production-path-audit-2026-09-13.md`,
`docs/reports/DS-R3-AR-surface-local-support-contract-2026-09-20.md`,
`AGENT_HANDOFF.md`, `agent swarm instantiation template/project/STATUS.md`.

Recorded (not recomputed here) current bindings from AGENT_COMMS/STATUS: graph
`75b6e9edf4ebc15fba3e7b1ec0bb1d3ecb8725c4dc3b3c9ee8946d49925a5568`, attribution
`ea5b97e3f8af0980b9f50b88f900ff09ca110c2d3ec254c17ea7fa8d28d6c01c`, matcher/solver
`9b19b63d60b4eb1309df7bd88251249ace9dec162ab66acb0a561446267d1d90`, runner
`c12a09fbfd783d6b6bc88729dfeadc3d664db18cd3186ffc9ac864d9c5bc5bbd`, entry
`5ced5b2baba3679516b7d75f668be433a2e58fe88ce56a6723e5abd118f60962`, lattice
`c782edbf60348f9307d387a61fcad5674394f8657d2147c0a03da5e49b16e673`, signatures
`86172efb7638d088b9bb8f44cf2c411ff9cd4cd3c2c6985cc6d668fcea69e245`.

Blocker: none for the audit. The recommended tranche needs a lead/Jev routing
decision and an Astra contract review of two points - the `.mcscan` schema version
(3 vs extend 2) and whether a geometry-only review volume is persisted - before any
implementation. No implementation, test, image or workload was performed. Stop.

## 7. Append-ready `AGENT_COMMS.md` entry

```text
### [2026-10-03] DeepSeek -> LEAD/Astra - DS-R4 read-only M4 production integration audit

Status: READ-ONLY AUDIT COMPLETE. No existing file edited, no test/image/workload run, no manifest or capture altered; all prior artifacts preserved. Full report: docs/reports/DS-R4-M4-production-integration-audit-2026-10-03.md (AGENT_COMMS.md itself is not valid UTF-8 at lines ~2280/3242/tail, so the read/edit tool could not append here without risking the file; the lead may paste this entry or the report).

Trace: cli.py image-derived-output/-deliver/export (L191-231, L387-448, L736-749) -> image_derived_production.derive_image_output_from_paths (L134) -> image_derived_entry.build_image_derived_entry (L388) [session via build_image_derived_session, SIFT connections via match_and_solve_pair, pose graph, then _persist_session L619 writes a session-only schema-2 ScanProject with an EMPTY BlockVolume] -> image_derived_lattice_measure.measure_image_derived_lattice (L819) -> image_derived_geometry_orchestration.derive_image_derived_geometry (L208; fixed gates radius 32, .90/.15/.02 at L249-253; stage g = image_derived_graph._derive_image_derived_graph L1233 -> plane_graph.match_plane_graph L1291 + plane_graph.solve_component L1297) -> image_derived_output.derive_image_derived_output (L296) -> _derive_ready (L232) placement/volume/face-observation/proposals/state_promotion.promote_plane_states (L994) -> image_derived_output_delivery.deliver_image_derived_output (L55; promoted + export_ready + exact_state_eligible + omitted_unknown==0) -> image_derived_delivery.deliver_image_derived_litematic -> LitematicExporter.export (L46).

Durable gaps: G1 the image-derived .mcscan never carries geometry - no image-derived module fills project.volume or persists the R1/R2-A records, so `export` on an image-derived project raises LitematicExportError("cannot export an empty volume") (litematic.py L59-60). G2 delivery requires a zero-omission volume: exact_state_eligible == (omitted_unknown == 0) (state_promotion.py L964/L988) with allow_omitted_unknown=False (image_derived_output_policy.py L185); the accepted V3 recipe has 15 unknown omissions (M4-V3-supported-volume-bridge-2026-09-19.md; M6-image-derived-orchestration-2026-09-19.md L255-266), so its promotion is ineligible. G3 real-image geometry never assembles: stage g abstains (v3b plane_integer_residual / first failed predicate minimum_resultant, resultants .39146/.24717/.24521 vs .90; L1B/R4 ambiguous_quadrant).

Accepted structural seams (code, not rendered-fixture proof): M7 entry, M9-C lattice, R1 geometry orchestration, R2-A/P output+policy, R2-D delivery, M4-V1/V2/V3, M5 face observations/proposals/promotion, schema-2 session persistence, M6 facade. No current end-to-end fixture reaches export/promotion: false.

Recommended single bounded next M4 production tranche - M4-V4b "persist and export the assembled image-derived production result": add a canonical digest-bound image_derived_production record (entry/lattice/geometry/output digests + review facts, no pixels) to image_derived_production.py; persist it as a new optional top-level ScanProject key through domain/models.py + project/store.py with schema-1/2 load preserved (Astra to choose schema 3 vs extended 2); on a promoted exact-state result also persist promotion.to_block_volume() into project.volume so the existing `export` writes it; on a non-promotable assembled result persist the typed review record and, only if the lead wants an artifact, the accepted plane_volume.to_geometry_only_preview(...) volume (INFERRED, zero confidence, exact_state_eligible=False, omitted cells absent); never relax allow_omitted_unknown or any gate. This is the remaining item of the accepted M4-V4 contract (M4-placement-volume-bridge-contract-2026-09-19.md section V4). Controls: schema round-trip byte-identical; promoted export block-for-block equal; review/refusal persists no fabricated blocks; tampered/forbidden/non-finite refused; no pixel payload and unchanged normal-mode signature; deterministic envelope; no matcher/solver/gate/capture change.

Alternatives rejected: solver/alias repair (measurement-gated; v4 run awaits user authorization), gate relaxation (forbidden), entry-gate decoupling (loses yaw-quadrant constraints), M6 facade CLI wiring (no output change), surface-local support (evaluation-only per DS-R3-AR).

No implementation, test, image or workload was performed. Stop for lead review and Jev routing.
```

## Lead erratum (2026-10-03): G1 / V4b persistence audit was stale

The DS-R4 G1 finding and its proposed V4b terminal-persistence tranche are superseded by current source. `image_derived_entry._persist_session` writes a schema-2 checkpoint, but this is intermediate: `image_derived_production.py::_result_envelope` binds a promotion and `to_block_volume()` volume for promoted results; `_publish` creates a schema-3 `ScanProject` carrying `image_derived_session`, `image_derived_production`, and volume, and calls `publish_project_no_clobber`. `domain/models.py` supports schema 3; `project/store.py::_canonical_production_record` replays the envelope against the session and volume; `cli.py::_export_schema3_project` routes a stored promoted envelope through verified `deliver_image_derived_litematic`. Do not implement the report's proposed duplicate envelope/schema/volume persistence.

The real unmet fact is narrower and more important: no current rendered fixture has reached promotion, so successful schema-3 export is not yet verified. G2's exact-state/zero-omission policy and G3's geometry abstentions remain blockers. The 2026-09-19 V4 contract additionally mentions richer review evidence (evidence blobs, alternatives, gauge/export transforms); this compact schema-3 terminal does not prove those optional review artifacts are persisted. Audit that delta separately only if needed; it is not the missing persistence mechanism alleged in G1.

Source checked: `src/mc_photogrammetry/pipeline/image_derived_production.py` (`_result_envelope` 171, `_publish` 199, production terminal `_publish` call paths 215 onward); `src/mc_photogrammetry/domain/models.py` (schema-3 version/keys 151-160); `src/mc_photogrammetry/project/store.py` (`_canonical_production_record` 177); `src/mc_photogrammetry/cli.py` (`_export_schema3_project` 305, export dispatch 789-790). This correction changes no implementation or threshold.