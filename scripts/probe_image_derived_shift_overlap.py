"""Output-neutral probe for fixed zero-shift support in plane matching."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


@dataclass(frozen=True, slots=True)
class ProbeResult:
    case: str
    known_shift: tuple[int, int]
    zero_shift_overlap: int
    shifted_support_overlap: int
    production_status: str
    production_reason: str | None
    selected_shift: tuple[int, int] | None
    fired_gate: str


_RECTIFICATION = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0, principal_point=(40.0, 40.0), centre=(0.0, 0.0))


def _reading(frame: str) -> lattice_planes.PlaneReading:
    return lattice_planes.PlaneReading(
        frame=frame, axis=0, side=-1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 5.0, 1.0),),
        period_pixels=8.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=4096,
        polygon=None, coverage_cells=1600, structural_share=0.9, confidence=0.9,
        rectification=_RECTIFICATION)


def _signature(frame: str, cells: np.ndarray, coverage: np.ndarray
               ) -> lattice_planes.PlaneSignature:
    return lattice_planes.PlaneSignature(
        reading=_reading(frame), blocks_per_period=1.0,
        cells=np.asarray(cells, dtype=np.float32),
        interior=np.zeros(cells.shape, dtype=np.float32),
        coverage=np.asarray(coverage, dtype=bool), origin_uv=(0.0, 0.0))


def _shifted_overlap(left: np.ndarray, right: np.ndarray,
                     shift: tuple[int, int]) -> int:
    """Count support where ``left[r+dr,c+dc]`` corresponds to ``right[r,c]``."""
    dr, dc = shift
    rows, columns = left.shape
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(columns, columns - dc)
    if r0 >= r1 or c0 >= c1:
        return 0
    return int(np.count_nonzero(
        left[r0 + dr:r1 + dr, c0 + dc:c1 + dc] & right[r0:r1, c0:c1]))


def build_fixture(case: str) -> tuple[lattice_planes.PlaneSignature,
                                      lattice_planes.PlaneSignature,
                                      tuple[int, int]]:
    if case not in ("zero_overlap", "fixed_whitening"):
        raise ValueError("unknown probe case")
    shape = (80, 80)
    rows, columns = np.indices(shape)
    # Periodic block texture plus a deterministic sparse nonperiodic cue.
    base = ((rows % 7) * 11 + (columns % 9) * 5).astype(np.float64) + 1.0
    base[17, 13] += 97.0
    roll = (0, 24 if case == "zero_overlap" else 20)
    known_shift = (-roll[0], -roll[1])
    left_coverage = np.zeros(shape, dtype=bool)
    left_coverage[10:70, 5:(25 if case == "zero_overlap" else 45)] = True
    right_coverage = np.roll(left_coverage, roll, axis=(0, 1))
    right_cells = np.roll(base, roll, axis=(0, 1))
    return (_signature("left", base, left_coverage),
            _signature("right", right_cells, right_coverage), known_shift)


def run_probe(case: str) -> ProbeResult:
    left, right, known_shift = build_fixture(case)
    zero_overlap = int(np.count_nonzero(left.coverage & right.coverage))
    truth_overlap = _shifted_overlap(left.coverage, right.coverage, known_shift)
    refusals: list[plane_graph.PlaneMatchRefusal] = []
    match = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32, refusals=refusals)
    reason = refusals[0].reason if refusals else None
    selected = match.cell_shift_rc if match is not None else None
    if reason == "insufficient_overlap" and truth_overlap >= plane_graph.MIN_OVERLAP_CELLS:
        gate = "zero_shift_overlap"
    elif selected != known_shift and truth_overlap >= plane_graph.MIN_OVERLAP_CELLS:
        gate = "fixed_whitening_mask"
    else:
        gate = "none"
    return ProbeResult(
        case=case, known_shift=known_shift,
        zero_shift_overlap=zero_overlap, shifted_support_overlap=truth_overlap,
        production_status="matched" if match is not None else "refused",
        production_reason=reason, selected_shift=selected, fired_gate=gate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("zero_overlap", "fixed_whitening"),
                        default="zero_overlap")
    arguments = parser.parse_args()
    print(json.dumps(asdict(run_probe(arguments.case)), sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
