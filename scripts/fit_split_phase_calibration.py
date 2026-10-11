"""Evaluation-only IO driver for the V2-B split-phase calibration harness.

This script is the **only** IO of the calibration package (prereg sections 1
and 5).  It is never wired into the product CLI and is not importable from the
package.  It offers three subcommands, one OS process each:

``synthesize``
    Pass 1a (label-bearing, evaluation-only oracle): synthesize the exact
    96-control / 864-observation schedule for ``--split``, write the redacted
    ``corpus_<split>.json`` and the sealed ``labels_<split>.json``, and print
    the sealed ``labels_sha256``.
``predict``
    Pass 1b (label-blind): accepts exactly the redacted corpus path plus
    ``--labels-sha256`` (a digest reference, never a label path or handle).  It
    calls the frozen P0 producer once per observation, freezes the complete
    per-tuple record, and writes ``predictions_<split>.json``.
``score``
    Pass 2 (label-using): starts only after the prediction bytes, digest, and
    ``frozen_at`` exist, then opens the sealed labels, scores the frozen bytes
    under D4, selects the lexicographically first feasible tuple, and writes
    exactly ``calibration.json`` plus ``score_audit_<split>.json``.

Every stage accepts only the exact frozen grid and the exact 96-control /
864-observation schedule; no artifact is ever written outside
``exports/calibration/v2b-p0/``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from mc_photogrammetry.reconstruction import split_phase_calibration as sp

ARTIFACT_DIR = Path("exports") / "calibration" / "v2b-p0"
EXPECTED_CONTROLS = 96
EXPECTED_OBSERVATIONS = 864


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(sp.canonical_json(record).decode("utf-8"))


def _authorized(path) -> Path:
    resolved = Path(path).resolve()
    root = ARTIFACT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(f"artifact path outside {ARTIFACT_DIR}: {path}")
    return resolved


def _require_frozen_schedule(corpus, name: str) -> None:
    if len(corpus.controls) != EXPECTED_CONTROLS \
            or len(corpus.observations) != EXPECTED_OBSERVATIONS:
        raise SystemExit(f"{name} is not the exact 96/864 frozen schedule")


def command_synthesize(args: argparse.Namespace) -> None:
    sigma = sp.SPLIT_PHASE_CALIBRATION_SIGMA[args.split]
    result = sp.synthesize_split(sigma)
    _require_frozen_schedule(result["corpus"], args.split)
    _write_json(ARTIFACT_DIR / f"corpus_{args.split}.json",
                result["corpus"].to_record())
    _write_json(ARTIFACT_DIR / f"labels_{args.split}.json",
                result["labels"].to_record())
    print(result["labels"].labels_sha256)


def command_predict(args: argparse.Namespace) -> None:
    corpus = sp.corpus_from_record(_read_json(_authorized(args.corpus)))
    if corpus.sigma != sp.SPLIT_PHASE_CALIBRATION_SIGMA[args.split]:
        raise SystemExit("corpus sigma does not match --split")
    if corpus.labels_sha256 != args.labels_sha256:
        raise SystemExit("corpus labels_sha256 does not match --labels-sha256")
    _require_frozen_schedule(corpus, args.split)
    grid = sp.FROZEN_GRID
    if not sp.validate_frozen_grid(grid):
        raise SystemExit("the frozen grid literal failed validation")
    outcome = sp.predict_corpus(
        corpus, grid, expected_grid_sha256=sp.FROZEN_GRID_SHA256)
    if outcome["status"] != "complete":
        raise SystemExit(f"prediction refused: {outcome}")
    frozen = sp.freeze_prediction_payload(
        outcome["payload"], frozen_at=_iso_now())
    _write_json(ARTIFACT_DIR / f"predictions_{args.split}.json", frozen)
    print(frozen["predictions_sha256"])


def command_score(args: argparse.Namespace) -> None:
    if args.split != "calibration":
        raise SystemExit("only the calibration split may be scored here")
    corpus = sp.corpus_from_record(_read_json(_authorized(args.corpus)))
    evaluation = sp.corpus_from_record(
        _read_json(_authorized(args.evaluation_corpus)))
    _require_frozen_schedule(corpus, "calibration corpus")
    _require_frozen_schedule(evaluation, "evaluation corpus")
    grid = sp.FROZEN_GRID
    if not sp.validate_frozen_grid(grid):
        raise SystemExit("the frozen grid literal failed validation")
    # The prediction bytes, digest, and frozen_at must exist before any label
    # is opened (prereg section 5 chronology).
    prediction = _read_json(_authorized(args.predictions))
    if "frozen_at" not in prediction \
            or not sp.verify_prediction_digest(prediction):
        raise SystemExit("prediction freeze is missing, tampered, or unfrozen")
    labels = sp.labels_from_record(_read_json(_authorized(args.labels)))
    if labels.labels_sha256 != corpus.labels_sha256:
        raise SystemExit("sealed labels_sha256 does not match the corpus")
    scored_at = _iso_now()
    outcome = sp.score_freeze(
        prediction, labels, grid, calibration_corpus=corpus,
        evaluation_corpus=evaluation)
    audit = sp.score_audit_payload(
        sigma=0, split=args.split, corpus_sha256=corpus.corpus_sha256,
        labels_sha256=labels.labels_sha256,
        predictions_sha256=prediction["predictions_sha256"],
        frozen_at=prediction["frozen_at"], scored_at=scored_at,
        outcome=outcome, grid_sha256=grid.grid_sha256, caps=sp.default_caps())
    if not audit["scored_after_frozen"]:
        raise SystemExit("score chronology violated: scored_at <= frozen_at")
    if outcome["status"] != "selected":
        raise SystemExit(f"no feasible calibration: {outcome}")
    _write_json(ARTIFACT_DIR / "calibration.json",
                outcome["calibration"].to_record())
    _write_json(ARTIFACT_DIR / f"score_audit_{args.split}.json", audit)
    print(outcome["calibration"].calibration_sha256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fit_split_phase_calibration.py",
        description="Evaluation-only V2-B split-phase calibration driver.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthesize = subparsers.add_parser("synthesize")
    synthesize.add_argument("--split", required=True,
                            choices=sp.SPLIT_PHASE_CALIBRATION_SPLITS)
    synthesize.set_defaults(handler=command_synthesize)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--split", required=True,
                         choices=sp.SPLIT_PHASE_CALIBRATION_SPLITS)
    predict.add_argument("--corpus", required=True)
    predict.add_argument("--labels-sha256", required=True)
    predict.set_defaults(handler=command_predict)

    score = subparsers.add_parser("score")
    score.add_argument("--split", required=True, choices=("calibration",))
    score.add_argument("--corpus", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--evaluation-corpus",
                       default=str(ARTIFACT_DIR / "corpus_evaluation.json"))
    score.set_defaults(handler=command_score)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
