"""Focused synthetic tests for the pure world/topology split admission contract.

All identities and digests below are synthetic. Nothing here reads data, runs a
capture, touches the factory, or authorizes training.
"""
from dataclasses import FrozenInstanceError, replace
from collections.abc import Mapping
import hashlib
import json

import pytest

from mc_photogrammetry.dataset import splits
from mc_photogrammetry.dataset import world_split_contract as contract
from mc_photogrammetry.dataset.schemas import SCHEMA_VERSION
from mc_photogrammetry.dataset.world_split_contract import (
    IDENTITY_SCOPE, INDEPENDENT_WORLD_ELIGIBILITY, POLICY, SCHEMA, SPLITS,
    AdmissionManifest, AdmissionRequest, LineageAlias, SourceScope, SplitComponent,
    SplitRow, WorldSplitContractError, WorldSplitFoldConflict, admit, replay,
    verify_replay)

F1 = "1" * 64
F2 = "2" * 64
F3 = "3" * 64
T1 = "a" * 64
T2 = "b" * 64
T3 = "c" * 64
ZERO = "0" * 64


_MISSING = object()


def make_row(row_id="r1", *, source="source-a", scene=_MISSING, fold="train",
             world="world-a", fingerprint=F1, topology=T1, group=None):
    # One source admits each scene identity once, so default the scene to the row
    # id; explicit scene values still exercise cross-row collisions.
    if scene is _MISSING:
        scene = row_id
    return SplitRow(row_id, source, scene, fold, world, fingerprint, topology,
                    splits.scene_group(topology) if group is None else group)


def make_request(rows, aliases=(), sources=None):
    if sources is None:
        grouped = {}
        for row in rows:
            grouped.setdefault(row.source_index, []).append(row.row_id)
        sources = tuple(SourceScope(name, tuple(sorted(ids)))
                        for name, ids in sorted(grouped.items()))
    return AdmissionRequest(SCHEMA, tuple(sources), tuple(rows), tuple(aliases))


def clone(record):
    """Deep-copy a manifest record so tampering cannot alias the original."""
    return json.loads(json.dumps(record))


def reseal(record):
    """Recompute a valid manifest digest after tampering.

    This mirrors the contract's canonical serialization exactly, so an adversarial
    record can be correctly resealed and still be required to refuse.
    """
    body = {key: value for key, value in record.items() if key != "manifest_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    record["manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return record


def derive_component_id(row_ids):
    """Independently recompute a component id for a resealed tamper record."""
    payload = {"schema": SCHEMA, "component": list(sorted(row_ids))}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def one_source_two_rows():
    """Two train rows under one source: valid record with two components."""
    rows = [
        make_row("r1", source="source-a", scene="scene-1", world="world-a",
                 topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-a", scene="scene-2", world="world-b",
                 topology=T2, fingerprint=F2, fold="train"),
    ]
    return rows, admit(make_request(rows)).to_record()


def shared_world_rows():
    """Two train rows joined by a shared world, so lineage closure is stable."""
    return [
        make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-a", topology=T2,
                 fingerprint=F2, fold="train"),
    ]


# ---------------------------------------------------------------------------
# Fold-separation behaviour
# ---------------------------------------------------------------------------

def test_same_world_different_topology_same_fold_is_admitted():
    rows = [make_row("r1", world="world-a", topology=T1, fold="train"),
            make_row("r2", world="world-a", topology=T2, fold="train")]
    manifest = admit(make_request(rows))
    assert manifest.row_count == 2
    assert manifest.component_count == 1
    assert manifest.components[0].fold == "train"
    assert manifest.components[0].row_ids == ("r1", "r2")


def test_same_world_different_topology_cross_fold_is_refused():
    rows = [make_row("r1", world="world-a", topology=T1, fold="train"),
            make_row("r2", world="world-a", topology=T2, fold="test")]
    with pytest.raises(WorldSplitFoldConflict) as error:
        admit(make_request(rows))
    assert error.value.row_ids == ("r1", "r2")
    assert error.value.folds == ("test", "train")


def test_same_topology_different_world_cross_fold_is_refused():
    rows = [make_row("r1", world="world-a", topology=T1, fold="train"),
            make_row("r2", source="source-b", world="world-b", topology=T1,
                     fingerprint=F2, fold="dev")]
    with pytest.raises(WorldSplitFoldConflict) as error:
        admit(make_request(rows))
    assert error.value.row_ids == ("r1", "r2")


def test_same_source_different_label_cross_fold_is_refused():
    rows = [make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F1, fold="dev")]
    with pytest.raises(WorldSplitFoldConflict):
        admit(make_request(rows))


def test_same_source_same_fold_recapture_is_admitted():
    rows = [make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F1, fold="train")]
    manifest = admit(make_request(rows))
    assert manifest.component_count == 1
    assert manifest.components[0].folds == ("train",)


def test_transitive_world_source_topology_bridge_refuses_cross_fold():
    rows = [
        make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-a", topology=T2,
                 fingerprint=F2, fold="train"),
        make_row("r3", source="source-c", world="world-b", topology=T2,
                 fingerprint=F3, fold="test"),
    ]
    with pytest.raises(WorldSplitFoldConflict) as error:
        admit(make_request(rows))
    assert error.value.row_ids == ("r1", "r2", "r3")


def test_transitive_bridge_same_fold_is_admitted_as_one_component():
    rows = [
        make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-a", topology=T2,
                 fingerprint=F2, fold="train"),
        make_row("r3", source="source-c", world="world-b", topology=T2,
                 fingerprint=F3, fold="train"),
    ]
    manifest = admit(make_request(rows))
    assert manifest.component_count == 1
    assert manifest.components[0].row_ids == ("r1", "r2", "r3")


def test_explicit_lineage_alias_bridges_changed_fingerprints():
    rows = [
        make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-b", topology=T2,
                 fingerprint=F2, fold="test"),
    ]
    alias = LineageAlias("lineage-1", "reviewed-lineage-note", ("r1", "r2"))
    with pytest.raises(WorldSplitFoldConflict):
        admit(make_request(rows, aliases=[alias]))
    same_fold = [rows[0], replace(rows[1], fold="train")]
    manifest = admit(make_request(same_fold, aliases=[alias]))
    assert manifest.component_count == 1
    assert manifest.components[0].folds == ("train",)


def test_lineage_alias_naming_unknown_or_shared_rows_refuses():
    rows = [make_row("r1"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F2)]
    with pytest.raises(WorldSplitContractError, match="unknown row"):
        make_request(rows, aliases=[LineageAlias("l1", "note", ("r1", "r9"))])
    with pytest.raises(WorldSplitContractError, match="multiple lineage"):
        make_request(rows, aliases=[LineageAlias("l1", "note", ("r1", "r2")),
                                    LineageAlias("l2", "note", ("r1", "r2"))])
    with pytest.raises(WorldSplitContractError):
        LineageAlias("l1", "note", ("r1",))


# ---------------------------------------------------------------------------
# Determinism, replay and tampering
# ---------------------------------------------------------------------------

def test_input_permutation_replays_identical_manifest():
    rows = [make_row("r1", source="source-a", world="world-a", topology=T1,
                     fingerprint=F1, fold="train"),
            make_row("r2", source="source-a", world="world-a", topology=T2,
                     fingerprint=F2, fold="train"),
            make_row("r3", source="source-b", world="world-b", topology=T3,
                     fingerprint=F3, fold="test")]
    sources = [SourceScope("source-a", ("r1", "r2")), SourceScope("source-b", ("r3",))]
    aliases = [LineageAlias("lineage-1", "note", ("r1", "r2"))]
    first = admit(make_request(rows, aliases=aliases, sources=sources))
    permuted_rows = [rows[2], rows[0], rows[1]]
    permuted_sources = [sources[1], sources[0]]
    permuted_aliases = [LineageAlias("lineage-1", "note", ("r2", "r1"))]
    second = admit(make_request(permuted_rows, aliases=permuted_aliases,
                                sources=permuted_sources))
    assert first.to_record() == second.to_record()
    assert first.manifest_sha256 == second.manifest_sha256
    assert replay(make_request(rows, aliases=aliases, sources=sources)) \
        .manifest_sha256 == first.manifest_sha256


def test_replay_recomputes_and_rejects_a_tampered_claim():
    rows = [make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train")]
    request = make_request(rows)
    manifest = admit(request)
    assert verify_replay(request, manifest.to_record()).to_record() == manifest.to_record()
    tampered = make_request([replace(rows[0], fold="dev")])
    with pytest.raises(WorldSplitContractError, match="does not match"):
        verify_replay(tampered, manifest.to_record())
    with pytest.raises(WorldSplitContractError):
        verify_replay(request, "not a manifest")


def test_tampered_manifest_bindings_refuse():
    rows = [make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train")]
    record = admit(make_request(rows)).to_record()

    digest_tampered = dict(record, manifest_sha256=ZERO)
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(digest_tampered)

    component_tampered = json.loads(json.dumps(record))
    component_tampered["components"][0]["component_id"] = ZERO
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(component_tampered)

    source_tampered = json.loads(json.dumps(record))
    source_tampered["sources"][0]["expected_row_ids"] = ["r9"]
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(source_tampered)

    authority_tampered = dict(record, training_authorized=True)
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(authority_tampered)


def test_tampered_scene_group_binding_refuses():
    with pytest.raises(WorldSplitContractError, match="scene_group"):
        make_row("r1", topology=T1, group=ZERO)


def test_manifest_round_trips_and_is_immutable():
    manifest = admit(make_request([make_row("r1")]))
    assert AdmissionManifest.from_record(manifest.to_record()) == manifest
    with pytest.raises(FrozenInstanceError):
        manifest.manifest_sha256 = ZERO
    with pytest.raises(FrozenInstanceError):
        manifest.rows[0].fold = "test"


# ---------------------------------------------------------------------------
# Closed coverage
# ---------------------------------------------------------------------------

def test_omitted_duplicate_foreign_and_undeclared_rows_refuse():
    scope = SourceScope("source-a", ("r1", "r2"))
    with pytest.raises(WorldSplitContractError, match="missing"):
        AdmissionRequest(SCHEMA, (scope,), (make_row("r1"),))
    with pytest.raises(WorldSplitContractError, match="duplicate row id"):
        AdmissionRequest(SCHEMA, (SourceScope("source-a", ("r1",)),),
                         (make_row("r1"), make_row("r1")))
    with pytest.raises(WorldSplitContractError, match="foreign"):
        AdmissionRequest(SCHEMA, (SourceScope("source-a", ("r1",)),),
                         (make_row("r1"), make_row("r2")))
    with pytest.raises(WorldSplitContractError, match="undeclared source"):
        AdmissionRequest(SCHEMA, (SourceScope("source-a", ("r1",)),),
                         (make_row("r1", source="source-b"),))
    with pytest.raises(WorldSplitContractError, match="duplicate declared source"):
        AdmissionRequest(SCHEMA, (SourceScope("source-a", ("r1",)),
                                  SourceScope("source-a", ("r2",))),
                         (make_row("r1"), make_row("r2")))


def test_source_scope_and_alias_shape_validation():
    with pytest.raises(WorldSplitContractError):
        SourceScope("source-a", ())
    with pytest.raises(WorldSplitContractError):
        SourceScope("source-a", ("r1", "r1"))
    with pytest.raises(WorldSplitContractError):
        SourceScope(None, ("r1",))
    with pytest.raises(WorldSplitContractError):
        SourceScope("source-a", ["r1"])


# ---------------------------------------------------------------------------
# Identity strictness and Unicode preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world", [
    None, True, 3, 3.5, "", "   ", " leading", "trailing ", "a\x00b", "a\nb",
    "\ud800", "x" * 300, "\t",
])
def test_invalid_world_identities_refuse_without_coercion(world):
    with pytest.raises(WorldSplitContractError):
        make_row("r1", world=world)


def test_invalid_row_and_scene_identities_refuse():
    with pytest.raises(WorldSplitContractError):
        make_row(None)
    with pytest.raises(WorldSplitContractError):
        make_row("r1", scene=7)
    with pytest.raises(WorldSplitContractError):
        make_row("r1", source="")
    with pytest.raises(WorldSplitContractError):
        make_row("r1", fold="holdout")


@pytest.mark.parametrize("field,value", [
    ("fingerprint", None),
    ("fingerprint", True),
    ("fingerprint", 7),
    ("fingerprint", "A" * 64),
    ("fingerprint", "0" * 63),
    ("fingerprint", "g" * 64),
])
def test_invalid_fingerprints_refuse(field, value):
    with pytest.raises(WorldSplitContractError):
        make_row("r1", **{field: value})


@pytest.mark.parametrize("topology", [None, True, 7, "0" * 63, "Z" * 64, ""])
def test_invalid_topology_digests_refuse(topology):
    with pytest.raises(WorldSplitContractError):
        make_row("r1", topology=topology, group=ZERO)


def test_valid_cjk_identity_is_preserved_exactly():
    row = make_row("r1", world="世界-城", scene="场景-一")
    assert row.world_identity == "世界-城"
    assert row.scene_id == "场景-一"
    manifest = admit(make_request([row]))
    assert manifest.rows[0].world_identity == "世界-城"
    assert manifest.to_record()["rows"][0]["world_identity"] == "世界-城"


def test_internal_spaces_and_case_are_not_normalized():
    row = make_row("r1", world="Canal Town", topology=T1, fingerprint=F1, fold="train")
    assert row.world_identity == "Canal Town"
    other = make_row("r2", source="source-b", world="canal town", topology=T2,
                     fingerprint=F2, fold="test")
    # Case folding or name normalization would merge these rows and refuse; the
    # strict policy preserves the declared labels and admits two components.
    manifest = admit(make_request([row, other]))
    assert manifest.component_count == 2


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def test_legacy_schema_shaped_no_world_input_stays_outside_new_admission():
    topology = T1
    legacy_entry = {
        "scene_id": "legacy-scene",
        "split": splits.split_for_topology(topology),
        "topology_digest": topology,
        "scene_group": splits.scene_group(topology),
        "provenance": {"staging_token": F1, "split_identity": "scene-topology"},
    }
    # The legacy shape is still readable and usable by the legacy split path.
    assert legacy_entry["split"] in SPLITS
    assert legacy_entry["split"] == splits.split_for_topology(topology)
    # The new namespace is deliberately not the legacy schema.
    assert SCHEMA != str(SCHEMA_VERSION)
    # New admission refuses the missing world identity instead of coercing it.
    with pytest.raises(WorldSplitContractError, match="world_identity"):
        SplitRow("legacy", "source-a", legacy_entry["scene_id"],
                 legacy_entry["split"],
                 legacy_entry["provenance"].get("world_identity"),
                 legacy_entry["provenance"]["staging_token"],
                 legacy_entry["topology_digest"], legacy_entry["scene_group"])


LEGACY_SALT = "mc-photogrammetry/dataset-split/v2"


def _reference_scene_group(topology):
    payload = {"salt": LEGACY_SALT, "group": "scene-topology", "topology": topology}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("topology", [T1, T2, T3, ZERO, "f" * 64])
def test_scene_group_and_legacy_split_values_are_unchanged(topology):
    assert splits.SPLIT_SALT == LEGACY_SALT
    assert splits.TRAIN_CUTOFF == 70 and splits.DEV_CUTOFF == 85
    reference = _reference_scene_group(topology)
    assert splits.scene_group(topology) == reference
    bucket = int(reference[:8], 16) % 100
    expected = "train" if bucket < 70 else "dev" if bucket < 85 else "test"
    assert splits.split_for_topology(topology) == expected
    # The new contract binds the frozen function rather than reimplementing it.
    assert make_row("r1", topology=topology).scene_group == reference


# ---------------------------------------------------------------------------
# Growth, limits and authority
# ---------------------------------------------------------------------------

def test_bridge_to_already_conflicting_frozen_split_refuses_without_mutation():
    base_rows = [
        make_row("r1", source="source-a", world="world-a", topology=T1,
                 fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-b", topology=T2,
                 fingerprint=F2, fold="test"),
    ]
    base = admit(make_request(base_rows))
    assert base.component_count == 2
    before = base.to_record()
    bridge = make_row("r3", source="source-a", world="world-a", topology=T2,
                      fingerprint=F3, fold="train")
    with pytest.raises(WorldSplitFoldConflict) as error:
        admit(make_request(base_rows + [bridge]))
    assert error.value.row_ids == ("r1", "r2", "r3")
    assert base.to_record() == before


def test_row_cap_exhaustion_yields_no_manifest(monkeypatch):
    monkeypatch.setattr(contract, "MAX_ROWS", 1)
    rows = [make_row("r1"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F2)]
    with pytest.raises(WorldSplitContractError):
        make_request(rows)


def test_source_cap_exhaustion_yields_no_manifest(monkeypatch):
    monkeypatch.setattr(contract, "MAX_SOURCES", 1)
    rows = [make_row("r1"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F2)]
    with pytest.raises(WorldSplitContractError):
        make_request(rows)


def test_component_cap_exhaustion_yields_no_manifest(monkeypatch):
    monkeypatch.setattr(contract, "MAX_COMPONENTS", 1)
    rows = [make_row("r1"),
            make_row("r2", source="source-b", world="world-b", topology=T2,
                     fingerprint=F2)]
    with pytest.raises(WorldSplitContractError, match="component"):
        admit(make_request(rows))


def test_canonical_byte_cap_exhaustion_yields_no_manifest(monkeypatch):
    monkeypatch.setattr(contract, "MAX_CANONICAL_BYTES", 16)
    with pytest.raises(WorldSplitContractError, match="byte cap"):
        admit(make_request([make_row("r1")]))


def test_two_world_all_train_is_structurally_admitted_but_unauthorized():
    rows = [
        make_row("r1", source="source-a", world="world-a", topology=T1,
                 fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-b", topology=T2,
                 fingerprint=F2, fold="train"),
    ]
    manifest = admit(make_request(rows))
    assert manifest.component_count == 2
    record = manifest.to_record()
    assert record["schema"] == SCHEMA
    assert record["policy"] == POLICY
    assert record["identity_scope"] == IDENTITY_SCOPE
    assert record["training_authorized"] is False
    assert record["independent_world_eligibility"] == INDEPENDENT_WORLD_ELIGIBILITY
    assert record["row_count"] == 2 and record["component_count"] == 2


# ---------------------------------------------------------------------------
# Pre-allocation caps and exact integer counts (DSCALE-32 blockers 1 and 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "source_count", "row_count", "component_count", "lineage_alias_count",
])
@pytest.mark.parametrize("value", [True, False, 1.0, 2.5, "1", None, -1])
def test_manifest_count_fields_require_exact_bounded_integers(field, value):
    record = admit(make_request([make_row("r1")])).to_record()
    record[field] = value
    with pytest.raises(WorldSplitContractError, match="exact integer|bounds"):
        AdmissionManifest.from_record(record)


@pytest.mark.parametrize("field,cap_name", [
    ("source_count", "MAX_SOURCES"),
    ("row_count", "MAX_ROWS"),
    ("component_count", "MAX_COMPONENTS"),
    ("lineage_alias_count", "MAX_LINEAGE_ALIASES"),
])
def test_manifest_over_cap_counts_refuse_before_parsing_arrays(monkeypatch, field,
                                                               cap_name):
    record = admit(make_request([make_row("r1")])).to_record()
    record[field] = 1
    monkeypatch.setattr(contract, cap_name, 0)
    with pytest.raises(WorldSplitContractError, match="bounds"):
        AdmissionManifest.from_record(record)


def test_manifest_over_cap_source_array_refuses_before_parsing_items(monkeypatch):
    record = admit(make_request([make_row("r1")])).to_record()
    monkeypatch.setattr(contract, "MAX_SOURCES", 1)
    # Counts stay consistent with the cap; only the list is oversized. A parser
    # that copied first would choke on the None items instead of the cap.
    record["sources"] = [None, None]
    record["source_count"] = 1
    with pytest.raises(WorldSplitContractError, match="cap"):
        AdmissionManifest.from_record(record)


def test_nested_membership_caps_refuse_before_identity_parsing(monkeypatch):
    record = admit(make_request([make_row("r1")])).to_record()
    monkeypatch.setattr(contract, "MAX_ROWS", 1)
    record["sources"][0]["expected_row_ids"] = [None, None]
    with pytest.raises(WorldSplitContractError, match="cap"):
        AdmissionManifest.from_record(record)

    record = admit(make_request([make_row("r1")])).to_record()
    record["lineage_aliases"] = [
        {"alias": "l1", "provenance": "note", "row_ids": [None, None]}]
    record["lineage_alias_count"] = 1
    with pytest.raises(WorldSplitContractError, match="cap"):
        AdmissionManifest.from_record(record)

    record = admit(make_request([make_row("r1")])).to_record()
    record["components"][0]["row_ids"] = [None, None]
    with pytest.raises(WorldSplitContractError, match="cap"):
        AdmissionManifest.from_record(record)

    record = admit(make_request([make_row("r1")])).to_record()
    record["components"][0]["folds"] = ["train", "dev", "test", "holdout"]
    with pytest.raises(WorldSplitContractError, match="cap"):
        AdmissionManifest.from_record(record)


def test_canonical_byte_cap_refuses_deserialization_without_manifest(monkeypatch):
    record = admit(make_request([make_row("r1")])).to_record()
    monkeypatch.setattr(contract, "MAX_CANONICAL_BYTES", 16)
    with pytest.raises(WorldSplitContractError, match="byte cap"):
        AdmissionManifest.from_record(record)


def test_lineage_alias_cap_exhaustion_refuses(monkeypatch):
    monkeypatch.setattr(contract, "MAX_LINEAGE_ALIASES", 1)
    with pytest.raises(WorldSplitContractError, match="alias cap"):
        make_request(shared_world_rows(),
                     aliases=[LineageAlias("l1", "note", ("r1", "r2")),
                              LineageAlias("l2", "note", ("r1", "r2"))])


# ---------------------------------------------------------------------------
# Lineage evidence retention (DSCALE-32 blocker 2)
# ---------------------------------------------------------------------------

def test_manifest_retains_lineage_alias_provenance_and_membership():
    alias = LineageAlias("lineage-1", "reviewed-note", ("r1", "r2"))
    request = make_request(shared_world_rows(), aliases=[alias])
    manifest = admit(request)
    record = manifest.to_record()
    assert record["lineage_alias_count"] == 1
    assert record["lineage_aliases"] == [
        {"alias": "lineage-1", "provenance": "reviewed-note",
         "row_ids": ["r1", "r2"]},
    ]
    assert AdmissionManifest.from_record(record) == manifest
    assert manifest.lineage_alias_count == 1
    assert replay(request).to_record() == record


def test_lineage_evidence_change_is_detected_even_with_identical_closure():
    alias = LineageAlias("lineage-1", "provenance-a", ("r1", "r2"))
    request = make_request(shared_world_rows(), aliases=[alias])
    manifest = admit(request)

    changed = clone(manifest.to_record())
    changed["lineage_aliases"][0]["provenance"] = "provenance-b"
    # A standalone resealed record that only changes declared provenance is still
    # internally consistent; replay against the original request detects it.
    resealed_change = reseal(clone(changed))
    assert AdmissionManifest.from_record(resealed_change) is not None
    with pytest.raises(WorldSplitContractError, match="does not match"):
        verify_replay(request, resealed_change)
    # Without resealing, the canonical digest catches the change.
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(changed)

    removed = clone(manifest.to_record())
    removed["lineage_aliases"] = []
    removed["lineage_alias_count"] = 0
    resealed_removed = reseal(clone(removed))
    assert AdmissionManifest.from_record(resealed_removed) is not None
    with pytest.raises(WorldSplitContractError, match="does not match"):
        verify_replay(request, resealed_removed)
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(removed)


def test_manifest_digest_binds_lineage_evidence():
    rows = shared_world_rows()
    first = admit(make_request(rows, aliases=[LineageAlias("l", "p1", ("r1", "r2"))]))
    second = admit(make_request(rows, aliases=[LineageAlias("l", "p2", ("r1", "r2"))]))
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.to_record() != second.to_record()


# ---------------------------------------------------------------------------
# Self-consistent deserialization of resealed records (DSCALE-32 blocker 3)
# ---------------------------------------------------------------------------

def test_resealed_source_coverage_tamper_refuses():
    _rows, record = one_source_two_rows()
    tampered = clone(record)
    tampered["sources"][0]["expected_row_ids"] = ["r1"]
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="foreign|missing"):
        AdmissionManifest.from_record(tampered)


def test_resealed_foreign_source_membership_tamper_refuses():
    _rows, record = one_source_two_rows()
    tampered = clone(record)
    tampered["sources"][0]["expected_row_ids"] = ["r1", "r2", "r9"]
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="missing"):
        AdmissionManifest.from_record(tampered)


def test_resealed_component_fold_disagreement_refuses():
    _rows, record = one_source_two_rows()
    tampered = clone(record)
    tampered["components"][0]["fold"] = "dev"
    tampered["components"][0]["folds"] = ["dev"]
    reseal(tampered)
    with pytest.raises(WorldSplitContractError):
        AdmissionManifest.from_record(tampered)


def test_resealed_row_fold_tamper_refuses():
    _rows, record = one_source_two_rows()
    tampered = clone(record)
    tampered["rows"][0]["fold"] = "dev"
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="closure"):
        AdmissionManifest.from_record(tampered)


def test_resealed_merged_closure_tamper_refuses():
    _rows, record = one_source_two_rows()
    tampered = clone(record)
    merged = ["r1", "r2"]
    tampered["components"] = [{
        "component_id": derive_component_id(merged),
        "row_ids": merged,
        "fold": "train",
        "folds": ["train"],
    }]
    tampered["component_count"] = 1
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="closure"):
        AdmissionManifest.from_record(tampered)


def test_resealed_split_closure_tamper_refuses():
    rows = [
        make_row("r1", world="world-a", topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", world="world-a", topology=T2,
                 fingerprint=F2, fold="train"),
    ]
    record = admit(make_request(rows)).to_record()
    assert record["component_count"] == 1
    tampered = clone(record)
    split_components = []
    for member in sorted(record["components"][0]["row_ids"]):
        split_components.append({
            "component_id": derive_component_id([member]),
            "row_ids": [member],
            "fold": "train",
            "folds": ["train"],
        })
    tampered["components"] = split_components
    tampered["component_count"] = 2
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="closure"):
        AdmissionManifest.from_record(tampered)


# ---------------------------------------------------------------------------
# Scene identity uniqueness per source (DSCALE-32 blocker 4)
# ---------------------------------------------------------------------------

def test_duplicate_source_scene_identity_refuses_in_request():
    rows = [
        make_row("r1", source="source-a", scene="scene-shared", world="world-a",
                 topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-a", scene="scene-shared", world="world-b",
                 topology=T2, fingerprint=F2, fold="train"),
    ]
    with pytest.raises(WorldSplitContractError, match="scene"):
        make_request(rows)


def test_same_scene_id_across_different_sources_is_allowed():
    rows = [
        make_row("r1", source="source-a", scene="scene-shared", world="world-a",
                 topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-b", scene="scene-shared", world="world-b",
                 topology=T2, fingerprint=F2, fold="train"),
    ]
    manifest = admit(make_request(rows))
    assert manifest.row_count == 2
    assert manifest.component_count == 2


def test_resealed_duplicate_scene_identity_refuses():
    rows = [
        make_row("r1", source="source-a", scene="scene-1", world="world-a",
                 topology=T1, fingerprint=F1, fold="train"),
        make_row("r2", source="source-a", scene="scene-2", world="world-b",
                 topology=T2, fingerprint=F2, fold="train"),
    ]
    record = admit(make_request(rows)).to_record()
    tampered = clone(record)
    tampered["rows"][1]["scene_id"] = tampered["rows"][0]["scene_id"]
    reseal(tampered)
    with pytest.raises(WorldSplitContractError, match="scene"):
        AdmissionManifest.from_record(tampered)


# ---------------------------------------------------------------------------
# Pre-allocation membership and record-key guards (DSCALE-33)
# ---------------------------------------------------------------------------

def test_component_membership_identity_is_bounded_before_hashing(monkeypatch):
    record = admit(make_request([make_row("r1")])).to_record()
    # A valid record still deserializes and hashes before any guard is armed.
    valid = SplitComponent.from_record(record["components"][0])
    assert valid.row_ids == ("r1",)
    assert valid.component_id == record["components"][0]["component_id"]

    calls = []

    def forbidden_component_hash(row_ids):
        calls.append(("component_hash", row_ids))
        raise AssertionError("component hash ran before membership validation")

    def forbidden_serialization(payload):
        calls.append(("serialize", payload))
        raise AssertionError(
            "canonical serialization ran before membership validation")

    monkeypatch.setattr(contract, "_component_id", forbidden_component_hash)
    monkeypatch.setattr(contract, "_canonical_bytes", forbidden_serialization)

    bad_ids = (
        "x" * (contract.MAX_IDENTITY_CHARS + 1),
        "a\x00b",
        "a\nb",
        "\ud800",
        " leading",
        "trailing ",
    )
    for bad in bad_ids:
        candidate = {
            "component_id": record["components"][0]["component_id"],
            "row_ids": [bad],
            "fold": "train",
            "folds": ["train"],
        }
        with pytest.raises(WorldSplitContractError):
            SplitComponent.from_record(candidate)

    # A valid first member followed by an invalid later member must refuse with
    # no hash or canonical serialization: the whole raw list is validated before
    # any member is copied, so an accepted prefix cannot leak into a tuple.
    mixed = {
        "component_id": record["components"][0]["component_id"],
        "row_ids": ["r1", "a\x00b"],
        "fold": "train",
        "folds": ["train"],
    }
    with pytest.raises(WorldSplitContractError):
        SplitComponent.from_record(mixed)
    assert calls == []


def test_direct_component_construction_bounds_membership_before_hashing(monkeypatch):
    calls = []

    def forbidden_component_hash(row_ids):
        calls.append(row_ids)
        raise AssertionError("component hash ran before membership validation")

    monkeypatch.setattr(contract, "_component_id", forbidden_component_hash)
    with pytest.raises(WorldSplitContractError):
        SplitComponent(ZERO, ("x" * (contract.MAX_IDENTITY_CHARS + 1),),
                       "train", ("train",))
    with pytest.raises(WorldSplitContractError):
        SplitComponent(ZERO, ("a\x00b",), "train", ("train",))
    assert calls == []


def test_component_record_round_trips_valid_unicode_members():
    rows = [make_row("世界", world="world-世界", scene="场景"),
            make_row("café", source="source-a", world="world-世界",
                     scene="场景-2")]
    record = admit(make_request(rows)).to_record()
    for item in record["components"]:
        rebuilt = SplitComponent.from_record(clone(item))
        assert rebuilt.to_record() == item
    assert [row["row_id"] for row in record["rows"]] == ["café", "世界"]


class _ExplodingKeysMapping(Mapping):
    """A foreign Mapping whose key access must never be reached."""

    def __init__(self, payload):
        self._payload = payload

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        raise AssertionError("key iteration ran before the schema guard")

    def __len__(self):
        raise AssertionError("__len__ ran before the schema guard")

    def keys(self):
        raise AssertionError("keys() ran before the schema guard")

    def __contains__(self, key):
        raise AssertionError("membership test ran before the schema guard")


class _OversizedDictSubclass(dict):
    """A dict subclass advertising an oversized, un-iterable key set."""

    def __iter__(self):
        raise AssertionError("key iteration ran before the schema guard")

    def keys(self):
        raise AssertionError("keys() ran before the schema guard")

    def __len__(self):
        return 10 ** 9


def test_record_keys_refuse_before_mapping_iteration_or_copy():
    record = admit(make_request([make_row("r1")])).to_record()
    # Exact built-in dict records keep round-tripping, including nested records.
    assert AdmissionManifest.from_record(record).to_record() == record
    assert SourceScope.from_record(record["sources"][0]).source_index == "source-a"

    with pytest.raises(WorldSplitContractError, match="exact built-in dict"):
        AdmissionManifest.from_record(_ExplodingKeysMapping(record))
    with pytest.raises(WorldSplitContractError, match="exact built-in dict"):
        SourceScope.from_record(_ExplodingKeysMapping(record["sources"][0]))
    with pytest.raises(WorldSplitContractError, match="exact built-in dict"):
        AdmissionManifest.from_record(_OversizedDictSubclass(record))


def test_record_key_sets_still_require_exact_cardinality():
    record = admit(make_request([make_row("r1")])).to_record()
    oversized = dict(record, extra_key="x")
    with pytest.raises(WorldSplitContractError, match="exactly"):
        AdmissionManifest.from_record(oversized)
    missing = {key: value for key, value in record.items() if key != "policy"}
    with pytest.raises(WorldSplitContractError, match="exactly"):
        AdmissionManifest.from_record(missing)
    # A foreign Mapping is refused before any key is copied or sorted.
    with pytest.raises(WorldSplitContractError):
        SplitComponent.from_record(_ExplodingKeysMapping(record["components"][0]))
