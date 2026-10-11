"""Synthetic inventory contracts only; no assets, inference, or truth data."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE, PaletteColor
from mc_photogrammetry.blocks import candidate_universe as subject
from mc_photogrammetry.blocks.candidate_universe import (
    AppearanceSupport,
    CandidateUniverse,
    InventoryCompleteness,
    LegalStateManifest,
    appearance_record,
    canonical_json,
    canonical_state_id,
)
from mc_photogrammetry.domain.models import BlockState


def make_universe(*, disposition="supported", completeness="exhaustive"):
    # Distinct legal property states outside v1 with intentionally identical RGB.
    states = (BlockState("minecraft:oak_log", (("axis", "x"),)),
              BlockState("minecraft:oak_log", (("axis", "z"),)))
    manifest = LegalStateManifest(
        minecraft_version="synthetic-1", source_manifest_sha256="1" * 64,
        manifest_id="synthetic-legal-states", generator="synthetic-enumerator",
        generator_version="1", provenance=("synthetic:fixture",),
        completeness=InventoryCompleteness(completeness))
    support = tuple(AppearanceSupport(
        state_id=canonical_state_id(state), disposition=disposition,
        exclusion_reasons=() if disposition == "supported" else ("multipart-unimplemented",),
        appearance=PaletteColor(state, (80, 80, 80)) if disposition == "supported" else None,
        model_record_sha256="2" * 64 if disposition == "supported" else None,
        texture_record_sha256="3" * 64 if disposition == "supported" else None)
        for state in states)
    return CandidateUniverse(
        minecraft_version="synthetic-1", source_archive_sha256="4" * 64,
        manifest=manifest, decoder_version="synthetic-decoder-1",
        support_version="synthetic-support-1", decoder_settings=(("variant", "declared"),),
        legal_states=states, appearance_support=support)


def test_hash_replay_is_canonical_and_independently_computable():
    universe = make_universe()
    record = universe.to_record()
    payload = {key: value for key, value in record.items() if key != "universe_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert universe.universe_sha256 == hashlib.sha256(encoded).hexdigest()
    decoded = json.loads(canonical_json(record))
    assert CandidateUniverse.from_record(decoded) == universe
    assert CandidateUniverse.from_record(dict(reversed(list(decoded.items())))) == universe
    assert make_universe().to_record() == record


def test_identical_appearances_preserve_distinct_property_states_outside_v1():
    universe = make_universe()
    assert len(universe.state_ids) == len(universe.supported_palette) == 2
    assert len(set(universe.legal_states)) == 2
    assert all(state not in {item.state for item in DEFAULT_PALETTE}
               for state in universe.legal_states)
    appearances = [appearance_record(item) for item in universe.supported_palette]
    assert appearances[0].pop("state") != appearances[1].pop("state")
    assert appearances[0] == appearances[1]
    assert universe.appearance_completeness == "exhaustive"
    assert universe.requires_abstention is False


def test_partial_support_never_relabels_full_legal_inventory_complete():
    universe = make_universe()
    unsupported = AppearanceSupport(universe.state_ids[1], "unsupported",
                                    ("uv-lock-unimplemented",))
    partial = dataclasses.replace(universe,
                                  appearance_support=(universe.appearance_support[0], unsupported))
    assert partial.legal_states == universe.legal_states
    assert partial.legal_inventory_sha256 == universe.legal_inventory_sha256
    assert len(partial.supported_palette) == 1
    assert partial.appearance_completeness == "unestablished"
    assert partial.requires_abstention
    assert partial.to_record()["appearance_support"][1]["exclusion_reasons"] == [
        "uv-lock-unimplemented"]
    assert CandidateUniverse.from_record(partial.to_record()) == partial


@pytest.mark.parametrize("disposition", ["unsupported", "unknown"])
def test_unknown_and_unsupported_support_require_reasons_and_abstention(disposition):
    universe = make_universe(disposition=disposition)
    assert universe.supported_palette == ()
    assert universe.appearance_completeness == "unestablished"
    assert universe.requires_abstention
    with pytest.raises(ValueError, match="require exclusion"):
        AppearanceSupport(universe.state_ids[0], disposition)


def test_unestablished_legal_inventory_and_named_model_declarations():
    universe = make_universe(completeness="unestablished")
    assert universe.requires_abstention
    certificate = InventoryCompleteness("certified_under_named_model", "synthetic-enumerator-v1")
    certified = dataclasses.replace(universe, manifest=dataclasses.replace(
        universe.manifest, completeness=certificate))
    assert certified.manifest.completeness.named_model == "synthetic-enumerator-v1"
    assert certified.requires_abstention
    assert CandidateUniverse.from_record(certified.to_record()) == certified
    with pytest.raises(ValueError):
        InventoryCompleteness("certified_under_named_model")
    with pytest.raises(ValueError):
        InventoryCompleteness("exhaustive", "model")
    with pytest.raises(ValueError):
        InventoryCompleteness("physical-truth-recall")


@pytest.mark.parametrize("label,value", [
    ("source_archive_sha256", "5" * 64), ("minecraft_version", "synthetic-2"),
    ("decoder_version", "synthetic-decoder-2"), ("support_version", "synthetic-support-2"),
    ("decoder_settings", (("variant", "other"),)),
])
def test_archive_version_decoder_and_settings_bind_identity(label, value):
    universe = make_universe()
    changes = {label: value}
    if label == "minecraft_version":
        changes["manifest"] = dataclasses.replace(universe.manifest, minecraft_version=value)
    assert dataclasses.replace(universe, **changes).universe_sha256 != universe.universe_sha256


@pytest.mark.parametrize("label,value", [
    ("source_manifest_sha256", "5" * 64), ("manifest_id", "other-manifest"),
    ("generator_version", "2"), ("provenance", ("synthetic:other",)),
])
def test_manifest_identity_and_provenance_bind_legal_inventory(label, value):
    universe = make_universe()
    changed = dataclasses.replace(universe, manifest=dataclasses.replace(
        universe.manifest, **{label: value}))
    assert changed.legal_inventory_sha256 != universe.legal_inventory_sha256
    assert changed.universe_sha256 != universe.universe_sha256


@pytest.mark.parametrize("label,value", [
    ("model_record_sha256", "5" * 64), ("texture_record_sha256", "5" * 64),
    ("appearance", PaletteColor(BlockState("minecraft:oak_log", (("axis", "x"),)),
                                (81, 80, 80))),
])
def test_full_appearance_model_and_texture_identities_are_bound(label, value):
    universe = make_universe()
    support = dataclasses.replace(universe.appearance_support[0], **{label: value})
    changed = dataclasses.replace(
        universe, appearance_support=(support, universe.appearance_support[1]))
    assert changed.appearance_inventory_sha256 != universe.appearance_inventory_sha256
    assert changed.universe_sha256 != universe.universe_sha256


@pytest.mark.parametrize("mutation", [
    "archive", "manifest", "appearance", "appearance_digest", "legal_digest", "support_digest",
    "universe_digest", "state_ids", "false_completeness", "missing_disposition", "extra_field",
])
def test_tamper_and_stale_digests_are_rejected(mutation):
    record = copy.deepcopy(make_universe().to_record())
    if mutation == "archive":
        record["source_archive_sha256"] = "5" * 64
    elif mutation == "manifest":
        record["manifest"]["source_manifest_sha256"] = "5" * 64
    elif mutation == "appearance":
        record["appearance_support"][0]["appearance"]["rgb"][0] += 1
    elif mutation == "appearance_digest":
        record["appearance_support"][0]["appearance_sha256"] = "0" * 64
    elif mutation in ("legal_digest", "support_digest", "universe_digest"):
        key = {"legal_digest": "legal_inventory_sha256",
               "support_digest": "appearance_inventory_sha256",
               "universe_digest": "universe_sha256"}[mutation]
        record[key] = "0" * 64
    elif mutation == "state_ids":
        record["state_ids"].reverse()
    elif mutation == "false_completeness":
        record["appearance_completeness"] = "unestablished"
    elif mutation == "missing_disposition":
        del record["appearance_support"][0]["disposition"]
    else:
        record["unrecognized"] = "ignored?"
    with pytest.raises(ValueError):
        CandidateUniverse.from_record(record)


@pytest.mark.parametrize("kind", ["duplicate", "reversed", "missing", "extra", "mutable"])
def test_inventory_and_support_cardinality_order_and_alias_rejection(kind):
    universe = make_universe()
    changes = {
        "duplicate": {"legal_states": (universe.legal_states[0],) * 2},
        "reversed": {"legal_states": tuple(reversed(universe.legal_states))},
        "missing": {"appearance_support": universe.appearance_support[:1]},
        "extra": {"appearance_support": universe.appearance_support * 2},
        "mutable": {"legal_states": list(universe.legal_states)},
    }
    with pytest.raises(ValueError):
        dataclasses.replace(universe, **changes[kind])


@pytest.mark.parametrize("state", [
    BlockState("Minecraft:stone"), BlockState("minecraft:stone", (("z", "a"), ("a", "b"))),
    BlockState("minecraft:stone", (("x", "a"), ("x", "b"))),
    BlockState("minecraft:oak_log", (("axis", "y"),)),
    BlockState("minecraft:stone", (("bad key", "yes"),)),
    BlockState("minecraft:stone", [["axis", "x"]]),
])
def test_noncanonical_states_are_rejected(state):
    with pytest.raises(ValueError):
        canonical_state_id(state)


@pytest.mark.parametrize("changes", [
    {"provenance": ()}, {"provenance": ["mutable"]}, {"provenance": (" ",)},
    {"provenance": ("ref", "ref")}, {"source_manifest_sha256": "A" * 64},
    {"generator": ""}, {"generator_version": "1\n2"},
])
def test_malformed_manifest_provenance_is_rejected(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(make_universe().manifest, **changes)


@pytest.mark.parametrize("changes", [
    {"rgb": [1, 2, 3]}, {"rgb": (True, 2, 3)}, {"rgb": (256, 2, 3)},
    {"descriptors": ((float("nan"),),)}, {"face_rgbs": ((1, 2),)},
    {"shape_boxes": (((0.0, 0.0, 0.0), [1.0, 1.0, 1.0]),)},
    {"model_faces": ({"mutable": "alias"},)}, {"full_cube": 1},
])
def test_mutable_or_malformed_appearance_payloads_are_rejected(changes):
    entry = make_universe().appearance_support[0]
    with pytest.raises(ValueError):
        dataclasses.replace(entry, appearance=dataclasses.replace(entry.appearance, **changes))


def test_supported_payload_requires_matching_state_and_all_record_digests():
    entry = make_universe().appearance_support[0]
    for changes in ({"model_record_sha256": None}, {"texture_record_sha256": "short"},
                    {"state_id": "minecraft:stone"}, {"exclusion_reasons": ("excluded",)},
                    {"disposition": "unsupported", "exclusion_reasons": ("excluded",)}):
        with pytest.raises(ValueError):
            dataclasses.replace(entry, **changes)


def test_returned_records_are_detached_and_frozen_payloads_reject_assignment():
    universe = make_universe()
    original = universe.to_record()
    record = universe.to_record()
    record["manifest"]["provenance"].append("new")
    record["appearance_support"][0]["appearance"]["rgb"][0] = 1
    assert universe.to_record() == original
    with pytest.raises(dataclasses.FrozenInstanceError):
        universe.decoder_version = "other"
    replay_record = json.loads(canonical_json(original))
    replay = CandidateUniverse.from_record(replay_record)
    replay_record["legal_states"][0]["properties"][0][1] = "z"
    assert replay.to_record() == original


@pytest.mark.parametrize("changes", [
    {"schema": "unknown"}, {"version": 1}, {"version": 3}, {"version": True},
    {"source_archive_sha256": "bad"}, {"decoder_settings": [["a", "b"]]},
    {"decoder_settings": (("a", "b"), ("a", "c"))},
])
def test_unknown_schema_version_and_invalid_universe_metadata(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(make_universe(), **changes)


def test_version_mismatch_and_forged_nested_digest_are_rejected():
    universe = make_universe()
    with pytest.raises(ValueError, match="version must match"):
        dataclasses.replace(universe, manifest=dataclasses.replace(
            universe.manifest, minecraft_version="other"))
    support = copy.copy(universe.appearance_support[0])
    object.__setattr__(support, "appearance_sha256", "0" * 64)
    with pytest.raises(ValueError, match="does not replay"):
        dataclasses.replace(universe, appearance_support=(support, universe.appearance_support[1]))


def test_all_appearance_fields_are_bound_and_mutable_nested_forgery_is_rejected():
    universe = make_universe()
    item = universe.appearance_support[0].appearance
    corner = (0.0, 0.0, 0.0)
    changes = {
        "face_rgbs": ((81, 80, 80),), "family": "synthetic-other",
        "descriptors": ((0.5,),), "face_map": (("up", (81, 80, 80)),),
        "face_thumbnails": (("up", "synthetic-thumbnail-record"),),
        "face_corners": (("up", (corner,) * 4),),
        "model_faces": (("up", (corner,) * 4, (0.0, 1.0, 0.0), None, None),),
        "full_cube": False, "shape_boxes": ((corner, (1.0, 1.0, 1.0)),),
        "covered_slots": ("up",),
    }
    for label, value in changes.items():
        palette = dataclasses.replace(item, **{label: value})
        support = dataclasses.replace(universe.appearance_support[0], appearance=palette)
        assert support.appearance_sha256 != universe.appearance_support[0].appearance_sha256
        assert AppearanceSupport.from_record(support.to_record()) == support
    manifest = copy.copy(universe.manifest)
    object.__setattr__(manifest, "provenance", ["synthetic:fixture"])
    with pytest.raises(ValueError):
        dataclasses.replace(universe, manifest=manifest)
    support = copy.copy(universe.appearance_support[0])
    object.__setattr__(support, "exclusion_reasons", [])
    with pytest.raises(ValueError):
        dataclasses.replace(universe, appearance_support=(support, universe.appearance_support[1]))


def forbid_work(*args, **kwargs):
    pytest.fail("unbounded input reached copying, sorting, decoding or hashing")


def nested_payload(depth, *, mutable=False):
    value = None
    for _ in range(depth):
        value = [value] if mutable else (value,)
    return value


def byte_ceiling(monkeypatch, ceiling):
    # Exercise the real preflight at a reduced test envelope while leaving the
    # production contract/digest declaration fixed. No large fixture is needed.
    original = subject._preflight

    def bounded(value, *, max_bytes=None, project=None):
        return original(value, max_bytes=ceiling if max_bytes is None else min(
            ceiling, max_bytes), project=project)

    monkeypatch.setattr(subject, "_preflight", bounded)


def test_versioned_resource_contract_is_identity_bound_and_refuses_tampering():
    universe = make_universe()
    record = universe.to_record()
    assert record["resource_contract"] == {
        "version": 1, "max_legal_states": 16384, "max_record_nodes": 2000000,
        "max_record_depth": 32, "max_string_characters": 65536,
        "max_metadata_bytes": 16000000, "max_canonical_bytes": 32000000,
        "max_integer_bits": 64}
    record["resource_contract"]["max_record_depth"] += 1
    with pytest.raises(ValueError, match="resource contract"):
        CandidateUniverse.from_record(record)


@pytest.mark.parametrize("entry", ["manifest", "appearance", "support"])
def test_real_string_boundary_on_standalone_and_nested_entries(monkeypatch, entry):
    universe = make_universe()
    text = "x" * subject.MAX_STRING_CHARACTERS
    if entry == "manifest":
        result = dataclasses.replace(universe.manifest, manifest_id=text)
        assert LegalStateManifest.from_record(result.to_record()) == result
        invoke = lambda: dataclasses.replace(result, manifest_id=text + "x")
    else:
        palette = dataclasses.replace(universe.appearance_support[0].appearance,
                                      model_faces=((text,),))
        result = dataclasses.replace(universe.appearance_support[0], appearance=palette)
        assert AppearanceSupport.from_record(result.to_record()) == result
        if entry == "appearance":
            invoke = lambda: appearance_record(dataclasses.replace(
                palette, model_faces=((text + "x",),)))
        else:
            invoke = lambda: dataclasses.replace(result, exclusion_reasons=(text + "x",))
    monkeypatch.setattr(subject, "_tuple_payload", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    monkeypatch.setattr(subject, "sorted", forbid_work, raising=False)
    with pytest.raises(ValueError, match="string resource"):
        invoke()


@pytest.mark.parametrize("entry", ["appearance", "support", "universe"])
def test_exact_depth_and_one_over_for_direct_copy_and_replay(monkeypatch, entry):
    universe = make_universe()
    support = universe.appearance_support[0]
    # model_faces is at depth 1, 2 or 4 in each complete record envelope.
    offset = {"appearance": 1, "support": 2, "universe": 4}[entry]
    palette = dataclasses.replace(support.appearance, model_faces=nested_payload(
        subject.MAX_RECORD_DEPTH - offset))
    if entry == "appearance":
        accepted = appearance_record(palette)
        subject._appearance_from_record(accepted)
        invoke = lambda: appearance_record(dataclasses.replace(palette, model_faces=(
            palette.model_faces,)))
        decode = subject._appearance_from_record
    else:
        supported = dataclasses.replace(support, appearance=palette)
        result = (supported if entry == "support" else dataclasses.replace(
            universe, appearance_support=(supported, universe.appearance_support[1])))
        accepted = result.to_record()
        assert type(result).from_record(accepted) == result
        invoke = lambda: dataclasses.replace(result, **(
            {"appearance": dataclasses.replace(palette, model_faces=(palette.model_faces,))}
            if entry == "support" else {"appearance_support": (
                copy_with_appearance(supported, dataclasses.replace(
                    palette, model_faces=(palette.model_faces,))),
                universe.appearance_support[1])}))
        decode = type(result).from_record
    # Extend the record without copying its deep payload recursively.
    bad = copy.deepcopy(accepted)
    appearance = (bad if entry == "appearance" else bad["appearance"] if entry == "support"
                  else bad["appearance_support"][0]["appearance"])
    appearance["model_faces"] = [appearance["model_faces"]]
    monkeypatch.setattr(subject, "_tuple_payload", forbid_work)
    monkeypatch.setattr(subject, "_decode_tuple", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    monkeypatch.setattr(subject, "sorted", forbid_work, raising=False)
    with pytest.raises(ValueError, match="node/depth"):
        invoke()
    with pytest.raises(ValueError, match="node/depth"):
        decode(bad)


def copy_with_appearance(support, appearance):
    # Forge a nested object so the outer constructor, rather than its nested
    # support constructor, is the API whose early validation is being tested.
    result = copy.copy(support)
    object.__setattr__(result, "appearance", appearance)
    return result


@pytest.mark.parametrize("entry", ["manifest", "support", "universe"])
def test_exact_complete_bytes_and_one_over_construction_serialization_replay(monkeypatch, entry):
    universe = make_universe()
    result = {"manifest": universe.manifest, "support": universe.appearance_support[0],
              "universe": universe}[entry]
    record = result.to_record()
    measured = len(canonical_json(record).encode("utf-8"))
    assert subject._preflight(result)[1] == measured
    byte_ceiling(monkeypatch, measured)
    assert dataclasses.replace(result).to_record() == record
    assert type(result).from_record(record) == result
    # Lowering the bound one byte makes this same full record one byte over.
    byte_ceiling(monkeypatch, measured - 1)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    monkeypatch.setattr(subject, "sorted", forbid_work, raising=False)
    monkeypatch.setattr(subject, "_appearance_from_record", forbid_work)
    monkeypatch.setattr(subject, "_state_from_record", forbid_work)
    for invoke in (lambda: dataclasses.replace(result), result.to_record,
                   lambda: type(result).from_record(record)):
        with pytest.raises(ValueError, match="byte/metadata"):
            invoke()


def test_aggregate_metadata_exact_and_one_over_precede_nested_decoding(monkeypatch):
    result = make_universe().manifest
    record = result.to_record()
    # Sum JSON string contents, including field names; no quote/separator bytes.
    metadata = sum(len(key) for key in record)
    metadata += sum(len(value) for value in record.values() if type(value) is str)
    metadata += sum(len(value) for value in result.provenance)
    metadata += sum(len(key) for key in record["completeness"])
    metadata += len(result.completeness.status)
    monkeypatch.setattr(subject, "MAX_METADATA_BYTES", metadata)
    assert LegalStateManifest.from_record(record) == result
    assert dataclasses.replace(result) == result
    monkeypatch.setattr(subject, "MAX_METADATA_BYTES", metadata - 1)
    monkeypatch.setattr(subject.InventoryCompleteness, "from_record", forbid_work)
    monkeypatch.setattr(subject, "sorted", forbid_work, raising=False)
    for invoke in (lambda: dataclasses.replace(result), result.to_record,
                   lambda: LegalStateManifest.from_record(record)):
        with pytest.raises(ValueError, match="byte/metadata"):
            invoke()


def test_exact_inventory_ceiling_and_atomic_excess_state_support_refusal(monkeypatch):
    template = make_universe(disposition="unknown")
    states = tuple(BlockState(f"minecraft:synthetic_{index:05}")
                   for index in range(subject.MAX_LEGAL_STATES))
    support = tuple(AppearanceSupport(state.name, "unknown", ("synthetic-only",))
                    for state in states)
    universe = dataclasses.replace(template, legal_states=states, appearance_support=support)
    assert len(universe.legal_states) == subject.MAX_LEGAL_STATES
    record = universe.to_record()
    assert CandidateUniverse.from_record(record) == universe
    monkeypatch.setattr(subject, "canonical_state_id", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    monkeypatch.setattr(subject, "_state_from_record", forbid_work)
    for changes in ({"legal_states": states + states[:1]},
                    {"appearance_support": support + support[:1]}):
        with pytest.raises(ValueError, match="count exceeds"):
            dataclasses.replace(universe, **changes)
    for key in ("legal_states", "appearance_support"):
        bad = {**record, key: record[key] + record[key][:1]}
        with pytest.raises(ValueError, match="count exceeds"):
            CandidateUniverse.from_record(bad)
    assert len(universe.legal_states) == subject.MAX_LEGAL_STATES


def test_exact_nodes_and_atomic_excess_appearance_container(monkeypatch):
    palette = make_universe().appearance_support[0].appearance
    palette = dataclasses.replace(palette, model_faces=((None,) * 100,))
    record = appearance_record(palette)
    nodes, _ = subject._preflight(palette)
    monkeypatch.setattr(subject, "MAX_RECORD_NODES", nodes)
    assert appearance_record(palette) == record
    assert subject._appearance_from_record(record) == palette
    monkeypatch.setattr(subject, "_tuple_payload", forbid_work)
    monkeypatch.setattr(subject, "_decode_tuple", forbid_work)
    larger = dataclasses.replace(palette, model_faces=(palette.model_faces[0] + (None,),))
    with pytest.raises(ValueError, match="node"):
        appearance_record(larger)
    record["model_faces"][0].append(None)
    with pytest.raises(ValueError, match="node"):
        subject._appearance_from_record(record)


@pytest.mark.parametrize("entry", ["appearance", "manifest", "support", "universe"])
def test_oversized_record_precedes_all_nested_decoders_and_serialization(monkeypatch, entry):
    universe = make_universe()
    result = {"appearance": universe.appearance_support[0].appearance,
              "manifest": universe.manifest, "support": universe.appearance_support[0],
              "universe": universe}[entry]
    record = appearance_record(result) if entry == "appearance" else result.to_record()
    record["extra-metadata"] = "x" * (subject.MAX_STRING_CHARACTERS + 1)
    decode = subject._appearance_from_record if entry == "appearance" else type(result).from_record
    monkeypatch.setattr(subject, "_decode_tuple", forbid_work)
    monkeypatch.setattr(subject, "_state_from_record", forbid_work)
    monkeypatch.setattr(subject, "record_sha256", forbid_work)
    monkeypatch.setattr(subject.json, "dumps", forbid_work)
    with pytest.raises(ValueError, match="string resource"):
        decode(record)


def test_cycles_alias_counts_and_huge_integer_refuse_before_json(monkeypatch):
    cycle = []
    cycle.append(cycle)
    shared = [None] * 100
    monkeypatch.setattr(subject.json, "dumps", forbid_work)
    with pytest.raises(ValueError, match="cycles"):
        canonical_json(cycle)
    with pytest.raises(ValueError, match="integer resource"):
        canonical_json(1 << 65)
    monkeypatch.setattr(subject, "MAX_RECORD_NODES", 203)
    assert subject._preflight([shared, shared])[0] == 203
    monkeypatch.setattr(subject, "MAX_RECORD_NODES", 202)
    with pytest.raises(ValueError, match="node"):
        canonical_json([shared, shared])


@pytest.mark.parametrize("value", [
    None, True, False, 0, -1, (1 << 64) - 1, -0.0, 1e20, 1e-7,
    "quote\"slash\\\b\f\n\r\t\x00\x1f\x7f\u00e9\U0001f600\ud800",
    {"z": [None, (), {"a": False}], "a": [1.0, "unicode-\u00e9"]},
])
def test_preflight_exact_json_byte_accounting_for_escaped_and_numeric_records(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert subject._preflight(value)[1] == len(encoded)
    assert subject._preflight(value, max_bytes=len(encoded))[1] == len(encoded)
    with pytest.raises(ValueError, match="byte/metadata"):
        subject._preflight(value, max_bytes=len(encoded) - 1)
