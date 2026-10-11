"""Deterministic scene-level train/dev/test splits.

Split membership is a pure function of a scene's *topology* - a digest of the
occupied coordinates and exact block states - and nothing else. It is never a
function of the capture seed, the camera sweep, or the view list. That is the
whole point: if a scene's split could depend on which cameras were used, then
recapturing an identical structure with a different camera seed could move it
into another split, and evaluation numbers would stop being comparable. Tying the
split to the topology also means two copies of the same structure always land in
the same split, so views of one structure can never straddle train and test.

The capture seed and recipe family are recorded alongside a scene for provenance
but deliberately do not enter the split: that is why the split identity is the
topology digest itself.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from mc_photogrammetry.dataset.hashing import digest_json
from mc_photogrammetry.dataset.schemas import DatasetSchemaError
from mc_photogrammetry.domain.models import BlockVolume

SPLITS = ("train", "dev", "test")
#: Cumulative buckets out of 100: 70% train, 15% dev, 15% test.
TRAIN_CUTOFF = 70
DEV_CUTOFF = 85
SPLIT_SALT = "mc-photogrammetry/dataset-split/v2"


def volume_topology(volume: BlockVolume) -> str:
    """A stable digest of a scene's occupied coordinates and block states.

    Deliberately excludes cameras, filenames, seeds, and timestamps: two captures
    of the same structure under different sweeps must hash identically.
    """
    rows = [
        [list(coordinate), observation.state.name,
         sorted((str(key), str(value)) for key, value in observation.state.properties)]
        for coordinate, observation in sorted(volume.items())
    ]
    return digest_json({"salt": SPLIT_SALT, "blocks": rows})


def topology_from_material(material: Any) -> str:
    """Topology digest for a non-volume description (recipe fields, block JSON)."""
    return digest_json({"salt": SPLIT_SALT, "material": material})


def scene_group(topology: str, *, topology_only: bool = True) -> str:
    """The stable scene/topology group digest recorded in the dataset index.

    Pure in the topology: two captures of one structure under different capture
    seeds share a group, so they cannot be split apart. ``topology_only=False``
    is not supported and exists only to make the contract explicit.
    """
    if not topology_only:
        raise DatasetSchemaError("a split group must be keyed by topology alone")
    if not topology:
        raise DatasetSchemaError("a scene needs a topology digest to be grouped")
    return digest_json({"salt": SPLIT_SALT, "group": "scene-topology",
                        "topology": str(topology)})


def topology_group(topology: str) -> str:
    """The split identity: the topology digest alone."""
    return scene_group(topology)


def scene_identity(seed: int, family: str, topology: str) -> str:
    """Provenance label (seed + family + topology). Never a split input."""
    del seed, family
    return scene_group(topology)


def assign_split(seed: int = 0, family: str = "custom", topology: str = "") -> str:
    """Map a scene's topology to one split, reproducibly, across processes.

    ``seed`` and ``family`` are accepted for call compatibility and deliberately
    ignored: the split must not change when a capture is reseeded.
    """
    del seed, family
    bucket = int(scene_group(topology)[:8], 16) % 100
    if bucket < TRAIN_CUTOFF:
        return "train"
    if bucket < DEV_CUTOFF:
        return "dev"
    return "test"


def split_for_topology(topology: str) -> str:
    """Split for a topology digest, with no reference to capture metadata."""
    return assign_split(topology=topology)


def split_for_scene(*, seed: int = 0, family: str = "custom",
                    volume: BlockVolume) -> str:
    """Split for a concrete scene, keyed by its block topology and never its views."""
    del seed, family
    return split_for_topology(volume_topology(volume))


def split_report(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check scene records for duplicate scene ids, groups, and split leakage.

    ``entries`` are finalization index entries. Returns counts per split; raises
    when the same scene id - or the same topology/scene group - appears with two
    different splits, which would mean one structure reached both training and
    evaluation. A group repeated inside a single split is recorded in
    ``repeated_groups``, not an error: recapturing a scene within its own split is
    allowed.
    """
    seen: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    references: dict[str, int] = {}
    counts = {split: 0 for split in SPLITS}
    for entry in entries:
        scene_id = str(entry.get("scene_id", ""))
        split = str(entry.get("split", ""))
        if split not in SPLITS:
            raise DatasetSchemaError(f"scene {scene_id!r} has unknown split {split!r}")
        if not scene_id:
            raise DatasetSchemaError("dataset index entry is missing a scene_id")
        previous = seen.get(scene_id)
        if previous is not None:
            raise DatasetSchemaError(
                f"scene {scene_id!r} appears in both {previous!r} and {split!r}")
        seen[scene_id] = split
        group = str(entry.get("scene_group") or "")
        if group:
            previous_split = seen_groups.get(group)
            if previous_split is not None and previous_split != split:
                raise DatasetSchemaError(
                    f"scene group {group!r} appears in both {previous_split!r} and "
                    f"{split!r}: identical topology must stay in one split")
            seen_groups.setdefault(group, split)
            references[group] = references.get(group, 0) + 1
        counts[split] += 1
    return {
        "scenes": len(seen),
        "groups": len(seen_groups),
        "repeated_groups": {group: count for group, count in sorted(references.items())
                            if count > 1},
        "by_split": counts,
    }
