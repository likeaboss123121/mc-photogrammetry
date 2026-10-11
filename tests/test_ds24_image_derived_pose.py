"""Controls for the DS-24 evaluation payload's per-variant serialization.

The DS-24 diagnostic builds ``evaluation.json`` after a CSV loop that binds a
``rows`` name to the *last* variant. The payload's ``per_variant`` block once
reused that leaked name, so every variant serialized the last variant's rows.

These controls pin the boundary:

* the payload-row helper reads ``variant_eval["rows"][variant]`` by name, so a
  synthetic fixture with deliberately unequal per-variant row counts serializes
  A/B/C with distinct counts and each variant's own ``selected`` values;
* the real diagnostic payload, once rerun, carries per-variant ``selected``
  counts equal to the frozen ``match_sets.json`` metadata (skipped cleanly when
  the DS-24 artifacts are not present in this checkout).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import diagnose_ds24_image_derived_pose as diagnostic  # noqa: E402

VARIANTS = diagnostic.VARIANTS


def _variant_eval_fixture():
    """A synthetic scorecard with unequal row counts per variant.

    The leaked-``rows`` regression would serialize the last variant's count (1)
    for every variant, so distinct counts here are the sentinel.
    """
    counts = {
        "A_reciprocal_ds14": 3,
        "B_directed_ds18": 2,
        "C_bestof_ratio_epipolar": 1,
    }
    rows: dict[str, list[dict]] = {}
    references: dict[int, dict] = {}
    fraction_by_slot: dict[str, dict] = {}
    for variant in VARIANTS:
        prefix = variant[0]
        rows[variant] = []
        for index in range(counts[variant]):
            slot = index
            rows[variant].append(
                {
                    "slot": slot,
                    "pair": f"left{slot}->right{slot}",
                    "selected": 100 * VARIANTS.index(variant) + index,
                    "decision": {
                        "solved": True,
                        "reason": "",
                        "right_yaw_class": VARIANTS.index(variant),
                        "inliers": 10 + index,
                        "inlier_fraction": 0.5,
                        "class_margin": 0.2,
                    },
                    "evaluation": {"class_equals_expected": True},
                    "usable": True,
                    "false_solve": False,
                }
            )
            references.setdefault(slot, {"reference_class": VARIANTS.index(variant)})
            fraction_by_slot.setdefault(str(slot), {})[f"{prefix}_selected"] = {
                "scorable": index,
                "true": index,
                "fraction": 0.0,
            }
    return {"rows": rows}, references, fraction_by_slot, counts


def test_payload_serializes_each_variants_own_rows_with_distinct_counts():
    """A/B/C must serialize their own row lists, not the leaked last variant."""
    variant_eval, references, fraction_by_slot, counts = _variant_eval_fixture()
    serialized = {
        variant: diagnostic.payload_variant_rows(
            variant, variant_eval, references, fraction_by_slot
        )
        for variant in VARIANTS
    }
    observed = {variant: len(rows) for variant, rows in serialized.items()}
    assert observed == counts
    assert len(set(observed.values())) == len(VARIANTS)
    for variant in VARIANTS:
        assert [row["selected"] for row in serialized[variant]] == [
            row["selected"] for row in variant_eval["rows"][variant]
        ]
        assert [row["slot"] for row in serialized[variant]] == list(range(counts[variant]))
        assert all(row["usable"] is True for row in serialized[variant])


def test_real_evaluation_selected_counts_match_match_sets_metadata():
    """After the unchanged rerun, the payload must mirror the frozen tape."""
    base = ROOT / "exports" / "diagnostics" / "ds24"
    evaluation_path = base / "evaluation.json"
    match_sets_path = base / "match_sets.json"
    if not (evaluation_path.exists() and match_sets_path.exists()):
        pytest.skip("DS-24 artifacts not present in this checkout")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    match_sets = json.loads(match_sets_path.read_text(encoding="utf-8"))
    by_slot = {int(pair["slot"]): pair for pair in match_sets["pairs"]}
    slots = sorted(by_slot)
    assert set(evaluation["per_variant"]) == set(VARIANTS)
    frozen_vectors = set()
    for variant in VARIANTS:
        expected = [
            int(by_slot[slot]["variants"][variant]["selected_correspondences"])
            for slot in slots
        ]
        rows = evaluation["per_variant"][variant]["rows"]
        assert [int(row["slot"]) for row in rows] == slots
        assert [int(row["selected"]) for row in rows] == expected
        frozen_vectors.add(tuple(expected))
    # The leaked-rows regression made all three variants serialize one vector.
    assert len(frozen_vectors) == len(VARIANTS)
