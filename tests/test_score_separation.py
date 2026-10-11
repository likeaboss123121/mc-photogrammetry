"""Focused contract tests for the diagnostic-only cue-separation script.

These tests pin the two behaviours the script's docstring and the acceptance
evaluator audit care about: correct tie handling in the AUC, and a verdict
partition that keeps the declining verdicts in the census while scoring only
the judged ones. They also pin the diagnostic-only domain label so the script
cannot silently be presented as a frozen usable-pixel acceptance metric.
"""
import numpy as np

from mc_photogrammetry.reconstruction.supported import (
    CONTRADICTED,
    FOREGROUND_OCCLUSION,
    OUTSIDE_IMAGE,
    SUPPORTED,
    WEAK_EVIDENCE,
)
from scripts.score_separation import (
    DIAGNOSTIC_DOMAIN,
    JUDGED_VERDICTS,
    _auc_with_ties,
    _average_ranks,
    _partition_judged,
)


def test_average_ranks_gives_ties_the_mean_position():
    ranks = _average_ranks(np.array([5.0, 5.0, 5.0]))
    assert list(ranks) == [2.0, 2.0, 2.0]


def test_average_ranks_handles_mixed_ties():
    ranks = _average_ranks(np.array([1.0, 3.0, 3.0, 9.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]


def test_auc_is_one_half_for_a_fully_tied_input():
    """The regression the docstring names: argsort ranking silently gave 0.0."""
    assert _auc_with_ties(np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0])) == 0.5


def test_auc_endpoints_and_partial_separation():
    assert _auc_with_ties(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert _auc_with_ties(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert _auc_with_ties(np.array([2.0, 4.0]), np.array([1.0, 3.0])) == 0.75


class _Cell:
    def __init__(self, cell, verdict, score):
        self.cell = cell
        self.verdict = verdict
        self.score = score


def test_partition_judged_keeps_every_verdict_but_scores_only_judged():
    cells = [
        _Cell((0, 0), SUPPORTED, 0.9),
        _Cell((0, 1), CONTRADICTED, 0.1),
        _Cell((0, 2), FOREGROUND_OCCLUSION, 0.5),
        _Cell((0, 3), WEAK_EVIDENCE, 0.2),
        _Cell((0, 4), OUTSIDE_IMAGE, 0.0),
    ]

    verdict_by_cell, cue_score = _partition_judged(cells)

    assert verdict_by_cell == {
        (0, 0): SUPPORTED,
        (0, 1): CONTRADICTED,
        (0, 2): FOREGROUND_OCCLUSION,
        (0, 3): WEAK_EVIDENCE,
        (0, 4): OUTSIDE_IMAGE,
    }
    assert cue_score == {(0, 0): 0.9, (0, 1): 0.1, (0, 2): 0.5}


def test_diagnostic_domain_is_labelled_and_verdicts_match_supported():
    assert "diagnostic-only" in DIAGNOSTIC_DOMAIN
    assert "never promotion evidence" in DIAGNOSTIC_DOMAIN
    assert set(JUDGED_VERDICTS) == {SUPPORTED, CONTRADICTED, FOREGROUND_OCCLUSION}
    assert WEAK_EVIDENCE not in JUDGED_VERDICTS
    assert OUTSIDE_IMAGE not in JUDGED_VERDICTS
