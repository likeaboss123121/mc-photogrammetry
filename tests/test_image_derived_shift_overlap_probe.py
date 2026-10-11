from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from scripts import probe_image_derived_shift_overlap as probe
from mc_photogrammetry.reconstruction import plane_graph


def test_zero_shift_overlap_now_admits_the_valid_known_shift():
    result = probe.run_probe("zero_overlap")
    assert result.known_shift == (0, -24)
    assert result.zero_shift_overlap == 0
    assert result.shifted_support_overlap == 1200
    assert result.production_status == "matched"
    assert result.production_reason is None
    assert result.selected_shift == result.known_shift
    assert result.fired_gate == "none"


def test_fixed_whitening_support_reports_selected_peak_against_truth():
    result = probe.run_probe("fixed_whitening")
    assert result.known_shift == (0, -20)
    assert result.zero_shift_overlap == 1200
    assert result.shifted_support_overlap == 2400
    assert result.production_status == "matched"
    assert result.selected_shift == result.known_shift
    assert result.fired_gate == "none"


@pytest.mark.parametrize("case", ("zero_overlap", "fixed_whitening"))
def test_probe_is_deterministic_and_fixture_order_is_output_neutral(case):
    first = probe.run_probe(case)
    probe.run_probe("fixed_whitening" if case == "zero_overlap" else "zero_overlap")
    assert probe.run_probe(case) == first
    left, right, known_shift = probe.build_fixture(case)
    left_cells, right_cells = left.cells.copy(), right.cells.copy()
    refusals = []
    reversed_match = plane_graph.match_plane_pair(
        right, left, quadrant=0, max_shift_cells=32, refusals=refusals)
    assert probe._shifted_overlap(
        right.coverage, left.coverage, tuple(-value for value in known_shift)) == \
        first.shifted_support_overlap
    assert reversed_match is not None
    assert reversed_match.cell_shift_rc == tuple(-value for value in known_shift)
    assert refusals == []
    np.testing.assert_array_equal(left.cells, left_cells)
    np.testing.assert_array_equal(right.cells, right_cells)


def test_cli_json_is_stable():
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/probe_image_derived_shift_overlap.py"),
               "--case", "zero_overlap"]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(root / "src") + (os.pathsep + existing if existing else "")
    kwargs = dict(check=True, capture_output=True, text=True,
                  cwd=root, env=environment)
    first = subprocess.run(command, **kwargs).stdout
    second = subprocess.run(command, **kwargs).stdout
    assert first == second
    assert json.loads(first)["fired_gate"] == "none"
