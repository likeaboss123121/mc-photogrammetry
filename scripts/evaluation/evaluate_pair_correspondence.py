"""One-shot corpus runner for the DS-R5I-R1 pair-correspondence scorecard.

Evaluation-only. It hash-pins every finalized ``exports/datasets/**/index.json``
entry, groups scenes strictly by ``scene_group``, runs the production SIFT
detector and mutual matcher over every unordered frame pair of every scene, scores
the returned match identities against the evaluator truth, and writes one sealed
JSON record. It edits no production source, threshold, matcher, solver, project
or promotion decision, and it refuses to overwrite an existing output.

DS-R5I-R1 correction: the appearance bucket is the exact-state/same-slot/
other-block proxy (no texture identity is read) and the truth denominator plus
returned identity are keyed by unique pixel pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from mc_photogrammetry.evaluation_pair_correspondence import (
    SCORECARD_SCHEMA,
    SURFACE_TOLERANCE_BLOCKS,
    DatasetSceneInput,
    PairScore,
    build_group_manifest,
    build_scene,
    check_fold_leakage,
    check_split_leakage,
    discover_finalized_entries,
    evaluate_scene,
    frame_record,
    geometry_policy_record,
    macro_average,
    metric_value,
    pair_record,
    seal_record,
    summarise_scene,
    totals_from_pairs,
)
from mc_photogrammetry.pipeline.correspondence import CorrespondenceSettings

DEFAULT_EXPORTS_ROOT = "exports/datasets"
DEFAULT_OUTPUT = ".tmp-pair-correspondence-eval-r1-20261003.json"

LIMITATIONS = (
    "Pair hypotheses are not truth correspondences; precision/recall are scored "
    "against an evaluator-built truth model, not against independent labels.",
    "Only three independent topology groups exist and every index split is "
    "'train', so leave-one-group-out is exploratory and no generalization claim "
    "is made.",
    "Partial block models (bottom/top slabs, stairs, trapdoors, fences, walls, "
    "panes, plants, fluids) are unavailable without the vanilla model assets; "
    "they occlude as full cubes but are excluded from truth ownership.",
    "The same-surface tolerance (one vanilla texel) is preregistered and not "
    "swept, so no sensitivity analysis is reported.",
    "No texture identity is available anywhere in the pipeline: the block texture "
    "is never read, hashed or compared. same_state_same_slot_other_block is only "
    "an appearance proxy (same exact BlockState, same face slot, different block "
    "coordinate) and is not texture ground truth.",
    "Truth pairs and returned identities are deduplicated by unique pixel pair; "
    "repeated descriptor keypoints at exactly the same pixel are one identity, "
    "and distinct pixels are never collapsed.",
    "One corpus and one run; no matcher, threshold, solver, production, project "
    "or promotion setting was changed.",
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", default=DEFAULT_EXPORTS_ROOT,
                        help="root containing finalized exports/datasets/*/index.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="sealed JSON output path (refuses to overwrite)")
    return parser.parse_args(list(argv))


def _group_record(
    scene_group: str,
    entries: Sequence[DatasetSceneInput],
    pairs: Sequence[PairScore],
) -> dict[str, Any]:
    totals = totals_from_pairs(pairs)
    return {
        "scene_group": scene_group,
        "scene_ids": sorted({entry.scene_id for entry in entries}),
        "datasets": sorted({entry.dataset_id for entry in entries}),
        "scene_entry_count": len(entries),
        "pair_count": len(pairs),
        "totals": {
            "pairs": totals.pairs,
            "returned": totals.returned,
            "positive": totals.positive,
            "same_face_out_of_tolerance": totals.same_face_out_of_tolerance,
            "same_state_same_slot_other_block": (
                totals.same_state_same_slot_other_block),
            "wrong_surface": totals.wrong_surface,
            "unattributed": totals.unattributed,
            "truth_pairs": totals.truth_pairs,
            "truth_pairs_returned": totals.truth_pairs_returned,
            "positive_identity_mismatch": (
                totals.positive - totals.truth_pairs_returned),
        },
        "precision": metric_value(totals.positive, totals.returned,
                                  undefined_reason="no_returned_matches"),
        "recall": metric_value(totals.truth_pairs_returned, totals.truth_pairs,
                               undefined_reason="no_truth_keypoint_pairs"),
    }


def build_payload(
    exports_root: str,
    entries: Sequence[DatasetSceneInput],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    settings = CorrespondenceSettings()
    scene_records: list[dict[str, Any]] = []
    pairs_by_group: dict[str, list[PairScore]] = {}
    all_pairs: list[PairScore] = []
    frames_total = 0
    keypoints_total = 0
    keypoints_with_truth_total = 0
    no_truth_totals: dict[str, int] = {}

    for entry in entries:
        scene = build_scene(entry)
        evaluation = evaluate_scene(scene, settings=settings)
        summary = summarise_scene(evaluation.pairs)
        frame_records = [frame_record(observation)
                         for observation in evaluation.observations]
        for record in frame_records:
            frames_total += 1
            keypoints_total += int(record["keypoints"])
            keypoints_with_truth_total += int(record["keypoints_with_truth"])
            for reason, count in record["no_truth_reasons"].items():
                no_truth_totals[reason] = no_truth_totals.get(reason, 0) + int(count)
        scene_records.append({
            "dataset_id": entry.dataset_id,
            "scene_id": entry.scene_id,
            "scene_group": entry.scene_group,
            "split": entry.split,
            "directory": entry.directory,
            "scene_root": entry.scene_root,
            "frame_count": len(evaluation.observations),
            "pair_count": len(evaluation.pairs),
            "frames": frame_records,
            "pairs": [pair_record(pair) for pair in evaluation.pairs],
            "summary": summary,
        })
        pairs_by_group.setdefault(entry.scene_group, []).extend(evaluation.pairs)
        all_pairs.extend(evaluation.pairs)

    group_records = [
        _group_record(group, [entry for entry in entries if entry.scene_group == group],
                      pairs_by_group.get(group, []))
        for group in sorted({entry.scene_group for entry in entries})
    ]
    pooled_totals = totals_from_pairs(all_pairs)
    pooled = {
        "totals": {
            "pairs": pooled_totals.pairs,
            "returned": pooled_totals.returned,
            "positive": pooled_totals.positive,
            "same_face_out_of_tolerance": pooled_totals.same_face_out_of_tolerance,
            "same_state_same_slot_other_block": (
                pooled_totals.same_state_same_slot_other_block),
            "wrong_surface": pooled_totals.wrong_surface,
            "unattributed": pooled_totals.unattributed,
            "truth_pairs": pooled_totals.truth_pairs,
            "truth_pairs_returned": pooled_totals.truth_pairs_returned,
            "positive_identity_mismatch": (
                pooled_totals.positive - pooled_totals.truth_pairs_returned),
        },
        "precision": metric_value(pooled_totals.positive, pooled_totals.returned,
                                  undefined_reason="no_returned_matches"),
        "recall": metric_value(pooled_totals.truth_pairs_returned,
                               pooled_totals.truth_pairs,
                               undefined_reason="no_truth_keypoint_pairs"),
    }
    macro = {
        "precision": macro_average([record["precision"]["value"]
                                    for record in group_records]),
        "recall": macro_average([record["recall"]["value"]
                                 for record in group_records]),
    }
    mismatch = pooled_totals.positive - pooled_totals.truth_pairs_returned
    payload: dict[str, Any] = {
        "schema": SCORECARD_SCHEMA,
        "status": "complete",
        "created_for": "DS-R5I-R1",
        "tool": "scripts/evaluation/evaluate_pair_correspondence.py",
        "parameters": {
            "correspondence_settings": {
                "max_features": settings.max_features,
                "ratio_test": settings.ratio_test,
                "min_matches": settings.min_matches,
                "min_keypoints_per_frame": settings.min_keypoints_per_frame,
                "spatial_grid": list(settings.spatial_grid),
                "min_spatial_coverage": settings.min_spatial_coverage,
                "min_mutual_fraction": settings.min_mutual_fraction,
                "mask_boundary_pad_px": settings.mask_boundary_pad_px,
            },
            "surface_tolerance_blocks": SURFACE_TOLERANCE_BLOCKS,
        },
        "inputs": {
            "exports_root": exports_root,
            "indexes": [
                {"dataset_id": entry.dataset_id, "index_path": entry.index_path,
                 "index_sha256": entry.index_sha256}
                for entry in entries
            ],
            "entries": [
                {
                    "dataset_id": entry.dataset_id,
                    "scene_id": entry.scene_id,
                    "scene_group": entry.scene_group,
                    "split": entry.split,
                    "directory": entry.directory,
                    "hashes": [list(item) for item in entry.hashes],
                    "screenshots": [list(item) for item in entry.screenshots],
                }
                for entry in entries
            ],
            "group_manifest": manifest,
            "split_leakage_issues": list(check_split_leakage(entries)),
            "fold_leakage_issues": list(check_fold_leakage(manifest)),
        },
        "scenes": scene_records,
        "groups": group_records,
        "pooled": pooled,
        "macro": macro,
        "renderer_geometry": {
            **geometry_policy_record(),
            "frames": frames_total,
            "keypoints": keypoints_total,
            "keypoints_with_truth": keypoints_with_truth_total,
            "keypoints_without_truth": keypoints_total - keypoints_with_truth_total,
            "no_truth_reason_totals": dict(sorted(no_truth_totals.items())),
        },
        "consistency": {
            "positive_identity_mismatch": mismatch,
            "ok": mismatch == 0,
        },
        "limitations": list(LIMITATIONS),
    }
    payload["seal_sha256"] = seal_record(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    if output.exists():
        print(f"refusing to overwrite {output}", file=sys.stderr)
        return 2
    try:
        entries = discover_finalized_entries(args.exports_root)
        manifest = build_group_manifest(entries)
        payload = build_payload(args.exports_root, entries, manifest)
    except Exception as error:  # noqa: BLE001 - one-shot runner reports and exits
        print(f"evaluation failed before writing {output}: {error}", file=sys.stderr)
        return 1
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    pooled = payload["pooled"]
    print(f"scenes={len(payload['scenes'])} groups={len(payload['groups'])} "
          f"frames={payload['renderer_geometry']['frames']} "
          f"pairs={pooled['totals']['pairs']}")
    print(f"pooled precision={pooled['precision']['value']} "
          f"recall={pooled['recall']['value']}")
    print(f"macro precision={payload['macro']['precision']['value']} "
          f"recall={payload['macro']['recall']['value']}")
    print(f"wrote {output} seal={payload['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
