# DS-36 — Batch settings propagation

Date: 2026-09-12. Scope: offline queue/settings contract only; no queue was materialized or executed.

## Contract

Batch requests without any explicit settings or pack field remain in deterministic `view_setting_strata` mode. Presence of `settings` (including `{}` or `null`) or either top-level pack field (including `null`) selects fixed mode. Fixed requests carry a strict `CaptureSettings` object to `build_plan` as `base_settings`; the CLI requires all four core keys (`gamma`, `time_of_day`, `weather`, `hud_hidden`) and accepts optional nullable pack keys. The mutually exclusive CLI options are `--fixed-settings-json` and `--stratified-settings`; omitting both preserves legacy stratified behavior.

Precedence is: task nested settings base, then variant nested keys override by key; task top-level pack fields override nested pack values, then variant top-level pack fields override task-level values. A present null top-level pack disables inherited pack selection. A present `settings: null` clears inherited nested settings to defaults, while any separate top-level pack field still applies. Settings JSON rejects unknown keys, coercive/wrong JSON types, invalid weather/ranges, and multiline/blank/non-string pack values. Pack names are carried as one JSON argv item, without shell interpolation. `PACK_NOTE` states that pack requests enter plan data but are not installed, selected, activated, or claimed active.

New manifests record `settings_mode`, `settings_requested`, and `settings_effective_request`. Queue loading cross-checks that declared fixed settings equal the prepare argv JSON, and that stratified requests have the matching strategy/view count. Legacy schema-v1 rows without the new fields remain loadable. Existing permissive `CaptureSettings.from_dict` remains unchanged for old serialized data; new batch/CLI inputs use the strict parser.

## Validation

- Focused batch settings/manifest cases: 14 passed.
- Focused prepare CLI fixed/stratified/invalid-input cases: 8 passed.
- Safe batch module: 47 passed; the sole child-process test `test_cli_materializes_a_unicode_identity_on_a_strict_cp1252_console` was deselected.
- Safe dataset module: 82 passed, 1 skipped; the sole child-process test `test_plan_is_identical_in_a_fresh_process` was deselected.
- `python -m py_compile` passed for both changed source modules, prepare script, and both test modules.

The prepare tests use temporary synthetic saves only and show fixed settings on every planned job, stratification with no settings flag, and invalid/contradictory fixed modes rejected before dataset creation. No queue, worker, client, or real capture was touched.

## Source SHA-256

- `src/mc_photogrammetry/dataset/schemas.py`: `4480cefaa23b2aa2d268e0b2125846e9c83dee44096bb8b2f93e3174a674d234`
- `src/mc_photogrammetry/dataset/batch.py`: `f61f75212913d3c926293cc8e35b82ed8141475d433ec796da67f8923db04900`
- `scripts/prepare_capture_dataset.py`: `83b8f587f9865f6139af59450ff9561a72d51a7ae4cd0776110241a506aa5c31`
- `tests/test_capture_batch.py`: `d0ba0c1513037c81fad211be1d8a612ca68937bfe0e2a74d4d359a68c4c50504`
- `tests/test_capture_dataset.py`: `78083ce7519028f19038584bf0af6fabb3bf09ea1f9603c93a1e5d11d77a6ae4`