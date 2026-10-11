"""Explicit post-freeze evaluator; not part of the normal CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mc_photogrammetry.evaluation_frozen_planes import evaluate_frozen_planes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--context-ground-truth", type=Path,
                        help="optional separately bound evaluation-only surrounding context")
    parser.add_argument("--vanilla-jar", required=True, type=Path)
    parser.add_argument("--palette-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = evaluate_frozen_planes(
        args.freeze, args.capture, args.ground_truth,
        args.vanilla_jar, args.palette_cache, args.output,
        context_ground_truth_path=args.context_ground_truth)
    print(json.dumps({"artifact_id": result["artifact_id"],
                      "output": args.output.name,
                      "frames": len(result["frames"]),
                      "micro_totals": result["micro_totals"],
                      "promotion_eligible": result["promotion"]["promotion_eligible"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
