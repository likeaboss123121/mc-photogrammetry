"""Focused A2 tests: the visible-denominator attribution contract.

This module tests the evaluation-only A2 implementation of
``docs/reports/image-only-loss-attribution-design-2026-10-03.md`` sections 4-6.
The tests are synthetic and self-contained: they read no fixture image, reference,
camera manifest or JAR, never import ``cli`` or production orchestration, and
produce no visible-truth number from fixture media. The tracked mask-specification
test reads only the reviewed evaluator JSON specifications for domain parity.

Coverage:

* module isolation (forbidden imports, IO, exporter and reconstruction calls);
* builder signatures carry no truth/reference/alignment/path parameter;
* the five labelled spaces and their fixed universes, never summed or interchanged;
* typed unavailable rows (`metric: null`, never a measured zero) for every stage
  after the first unreachable one, for both routes;
* the primary exact-state denominator is all visible ``V`` with ``V inter palette``
  only as a separate conditional row;
* alignment is integer translation plus lattice Y-quarter-turn only, and the
  scorer agrees with ``evaluation.visible_metrics`` for every rotation; a
  run-native palette receives the same directional state rotation as the candidate
  (translation cannot touch a coordinate-free palette), so palette coverage, the
  in-palette conditional denominator and the candidate/E rows stay consistent while
  an out-of-palette truth state keeps the full ``|V|`` denominator;
* non-monotone, in-universe transition reconciliation with a disjoint
  ``geometry_lost``/``state_lost`` partition on every Route B E-to-E step, while the
  G-to-G ``placed -> volume_cell`` step stays unpartitioned;
* checkpoint-reach gating: a digest-bound entry aggregate is never scored when the
  frontier says entry was unreached, and Route B promotion/export substages never
  inherit the coarse A1 ``output`` reach (``null`` when the record is missing,
  ``true`` from explicit record presence - including a carried empty record);
* candidate availability ceilings and the out-of-palette unreachable case;
* deterministic canonical serialization and digest binding;
* a synthetic recipe that exercises the real public volume/proposal/promotion
  constructors as an A2 route-B tail record;
* the tracked mask-specification contract: JSON specifications are loaded with
  the accepted evaluator rectangle/polygon semantics, the three tracked
  ``docs/evaluation_masks/templates-2`` specifications reproduce the reviewed
  pinned ``domain_sha256`` values, and a pinned run fails closed on any
  mask-domain divergence instead of scoring it.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_image_only_attribution as attribution
from mc_photogrammetry.blocks.appearance import PaletteColor
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation import visible_metrics
from mc_photogrammetry.evaluation_image_only_frontier import (
    route_a_frontier,
    route_b_frontier,
)
from mc_photogrammetry.reconstruction import appearance_state_proposals as proposals_mod
from mc_photogrammetry.reconstruction import plane_placement as placement_mod
from mc_photogrammetry.reconstruction import plane_volume as volume_mod
from mc_photogrammetry.reconstruction import state_promotion as promotion_mod
from mc_photogrammetry.reconstruction.face_observation import (
    NORMAL_OF_SLOT,
    FaceObservation,
)

SOURCE = Path(attribution.__file__)
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "evaluation" / "report_image_only_attribution.py"

FORBIDDEN_SIGNATURE_TOKENS = (
    "path", "manifest", "reference", "truth", "alignment", "coordinate", "bounds",
)
FORBIDDEN_SOURCE_IMPORTS = (
    "mc_photogrammetry.cli", "mc_photogrammetry.project",
    "mc_photogrammetry.pipeline", "mc_photogrammetry.export",
    "mc_photogrammetry.reconstruction", "pathlib", "os", "tempfile", "subprocess",
)
FORBIDDEN_SOURCE_TOKENS = (
    "LitematicExporter", "load_litematic", "load_image", "image_derived_production",
    "argparse", "subprocess", "resolve_face_placement", "build_plane_volume",
    "propose_face_states", "promote_plane_states", "open", "Path",
)

STONE = BlockState("minecraft:stone")
DIRT = BlockState("minecraft:dirt")
GOLD = BlockState("minecraft:gold_block")
STAIRS_NORTH = BlockState("minecraft:oak_stairs", (("facing", "north"),))
STAIRS_EAST = BlockState("minecraft:oak_stairs", (("facing", "east"),))
_SENTINEL = object()


def _visible_reference(reference, pixels=4):
    return attribution.visible_reference_from_counts(
        reference, {coordinate: pixels for coordinate in reference},
        source="synthetic")


def _alignment(turns=0, translation=(0, 0, 0)):
    return attribution.alignment_record(
        rotation_quarter_turns=turns, translation=translation,
        source="synthetic")


def _score(reference, candidate, *, visible=None, turns=0, translation=(0, 0, 0),
           space=attribution.MetricSpace.GEOMETRY, palette=None,
           checkpoint="facade_grid"):
    visible_reference = visible or _visible_reference(reference)
    return attribution.score_aligned_cells(
        reference=reference, candidate=candidate, visible_reference=visible_reference,
        alignment=_alignment(turns, translation), checkpoint=checkpoint,
        space=space, palette_states=palette)


def _unrot(coordinate, turns):
    x, y, z = coordinate
    for _ in range((-turns) % 4):
        x, z = -z, x
    return x, y, z


def _inverse_translation(translation, turns):
    # t maps candidate -> reference as R(c) + t; the inverse translation is
    # -R^{-1}(t), because R is not translation-invariant for a rotating frame.
    return tuple(-value for value in _unrot(translation, turns))


def _candidate_aligned_to(reference, turns, translation):
    """A candidate whose aligned form is exactly the reference set."""
    inverse = _inverse_translation(translation, turns)
    return {
        tuple(_unrot(coordinate, turns)[axis] + inverse[axis] for axis in range(3)):
            state
        for coordinate, state in reference.items()
    }


def _load_runner(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------

def test_module_has_no_io_no_production_imports_and_no_reconstruction_calls():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in FORBIDDEN_SOURCE_IMPORTS:
        assert not any(name == forbidden or name.startswith(forbidden + ".")
                       for name in imported), (forbidden, imported)
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.FunctionDef):
            identifiers.add(node.name)
        elif isinstance(node, ast.ClassDef):
            identifiers.add(node.name)
    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in identifiers, token


def test_builders_take_no_truth_path_manifest_or_coordinate_parameter():
    """A2 may take loaded evaluator objects, but never a path, manifest or bounds."""
    path_tokens = ("path", "manifest", "coordinate", "bounds", "file", "directory")
    for name in ("route_a_attribution", "route_b_attribution", "build_attribution",
                 "score_aligned_cells", "align_candidate",
                 "visible_reference_from_counts", "palette_coverage"):
        signature = inspect.signature(getattr(attribution, name))
        for parameter in signature.parameters:
            lowered = parameter.lower()
            assert not any(token in lowered for token in path_tokens), (name, parameter)
    # The evaluator-only builders are *permitted* already-loaded reference and
    # alignment objects; production still receives none of them.
    for name in ("route_a_attribution", "route_b_attribution"):
        parameters = set(inspect.signature(getattr(attribution, name)).parameters)
        assert {"reference", "alignment", "visible_reference"} <= parameters
    # The Route A builder may additionally take the run's own production records.
    route_a = set(inspect.signature(attribution.route_a_attribution).parameters)
    assert {"facade", "palette", "palette_terminal"} <= route_a


def test_runner_script_imports_and_exposes_run_and_main():
    assert RUNNER.is_file()
    module = _load_runner("_a2_report_runner", RUNNER)
    assert callable(module.main)
    assert callable(module.run_route_a)
    assert callable(module.run_route_b)
    assert callable(module.load_frozen_truth)
    assert callable(module.route_b_evaluator_pair_control)
    assert module.ATTRIBUTION_PRODUCER == attribution.ATTRIBUTION_PRODUCER


def test_runner_refuses_a_scale_or_reflection_alignment_field():
    module = _load_runner("_a2_report_runner_alignment", RUNNER)
    # The accepted alignment computation accepts valid integer inputs ...
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT}
    computed = module.compute_alignment(
        reference, dict(reference), _visible_reference(reference))
    assert computed["rotation_quarter_turns"] == 0
    assert computed["transform"] == attribution.ALIGNMENT_TRANSFORM
    # ... and a record carrying a scale/reflection/per-cell field fails closed.
    with pytest.raises(attribution.AttributionRefusal):
        attribution.alignment_from_record({
            "rotation_quarter_turns": 0, "translation": [0, 0, 0],
            "transform": attribution.ALIGNMENT_TRANSFORM, "scale": 2.0})
    with pytest.raises(attribution.AttributionRefusal):
        attribution.alignment_from_record({
            "rotation_quarter_turns": 0, "translation": [0, 0, 0],
            "transform": "candidate rotated then scaled to fit"})


# ---------------------------------------------------------------------------
# Labelled spaces and fixed universes
# ---------------------------------------------------------------------------

def test_five_labelled_spaces_keep_their_fixed_universes():
    assert {space.value for space in attribution.MetricSpace} == {
        "G", "O", "P", "C", "E", "camera"}
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.GEOMETRY] == \
        attribution.REFERENCE_CELLS_UNIVERSE
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.CANDIDATE] == \
        attribution.REFERENCE_CELLS_UNIVERSE
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.EXACT_STATE] == \
        attribution.REFERENCE_CELLS_UNIVERSE
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.OWNERSHIP] == \
        attribution.FRAME_PIXELS_UNIVERSE
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.PAIR_CORRESPONDENCE] == \
        attribution.PIXEL_PAIRS_UNIVERSE
    assert attribution.UNIVERSE_OF_SPACE[attribution.MetricSpace.CAMERA_DIAGNOSTIC] == "none"


def test_a_section_refuses_a_row_or_transition_from_another_space():
    row = attribution.AttributionRow(
        checkpoint="facade_grid", metric="geometry_recall",
        space=attribution.MetricSpace.GEOMETRY, kind="geometry",
        universe=attribution.REFERENCE_CELLS_UNIVERSE,
        denominator="all frozen visible V", reached=False, typed_reason="x")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionSection(
            space=attribution.MetricSpace.EXACT_STATE,
            universe=attribution.REFERENCE_CELLS_UNIVERSE, rows=(row,))


def test_a_camera_diagnostic_has_no_denominator_universe_or_transition():
    row = attribution.AttributionRow(
        checkpoint="camera", metric="camera_diagnostic",
        space=attribution.MetricSpace.CAMERA_DIAGNOSTIC, kind="camera",
        universe="none", denominator="none", reached=True,
        metric_value={"value": None, "numerator": None, "denominator": None,
                      "undefined_reason": None})
    assert row.universe == "none"
    assert row.denominator == "none"
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionTransition(
            universe="none", previous_checkpoint="facade_grid",
            next_checkpoint="camera", space=attribution.MetricSpace.CAMERA_DIAGNOSTIC,
            previous=1, gained=0, lost=1, net=-1)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionRow(
            checkpoint="camera", metric="camera_diagnostic",
            space=attribution.MetricSpace.CAMERA_DIAGNOSTIC, kind="camera",
            universe="none", denominator="frozen_reference_cells", reached=True,
            metric_value={"value": None, "numerator": None, "denominator": None,
                          "undefined_reason": None})


# ---------------------------------------------------------------------------
# Scoring and the permitted alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("turns,translation", [
    (0, (0, 0, 0)), (1, (2, 3, -1)), (2, (-4, 1, 5)), (3, (0, -2, 2)),
])
def test_scorer_agrees_with_the_accepted_evaluator_for_every_rotation(turns, translation):
    reference = {(x, 0, 0): (STONE if x % 2 == 0 else DIRT) for x in range(6)}
    visible_reference = _visible_reference(reference)
    candidate = _candidate_aligned_to(visible_reference.visible, turns, translation)
    score = attribution.score_aligned_cells(
        reference=reference, candidate=candidate, visible_reference=visible_reference,
        alignment=_alignment(turns, translation), checkpoint="facade_grid",
        space=attribution.MetricSpace.GEOMETRY)
    expected = visible_metrics(
        reference, candidate, set(visible_reference.visible),
        translation=translation, rotation_quarter_turns=turns)
    assert len(score.matched) == expected.visible_coordinate_matches
    assert len(score.exact) == expected.visible_exact_state_matches
    assert score.visible_cells == expected.visible_reference_blocks
    assert len(score.matched) == len(visible_reference.visible)
    assert len(score.exact) == len(visible_reference.visible)


def test_a_candidate_that_matches_the_reference_scores_full_geometry_and_state():
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): STONE}
    score = _score(reference, dict(reference))
    assert score.geometry_recall["value"] == 1.0
    assert score.exact_state_over_visible["value"] == 1.0
    assert score.phantom_cells == 0


def test_a_misaligned_candidate_scores_zero_overlap_and_no_phantoms():
    reference = {(x, 0, 0): STONE for x in range(4)}
    candidate = {(x, 50, 0): STONE for x in range(4)}
    score = _score(reference, candidate)
    assert len(score.matched) == 0
    assert score.geometry_recall["value"] == 0.0
    # Precision over a zero placed denominator is a typed null, never 0.0: the
    # candidate placed no cell inside the reference, so the denominator is empty.
    assert score.geometry_precision["value"] == 0.0
    assert score.geometry_precision["denominator"] == 4
    # Every placed candidate is outside the reference, so all four are phantoms.
    assert score.phantom_cells == 4
    # An empty candidate has an empty placed denominator: a typed null, never 0.0.
    empty = _score(reference, {})
    assert empty.geometry_recall["value"] == 0.0
    assert empty.geometry_precision["value"] is None
    assert empty.geometry_precision["undefined_reason"] == "no_placed_candidate_cells"


def test_alignment_record_rejects_scale_fit_reflection_and_bad_rotation():
    with pytest.raises(attribution.AttributionRefusal):
        attribution.alignment_record(rotation_quarter_turns=4, translation=(0, 0, 0),
                                     source="x")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.alignment_record(rotation_quarter_turns=0, translation=(0, 0),
                                     source="x")
    for forbidden in ("scale", "scale_fit", "reflection", "per_cell", "stretch"):
        with pytest.raises(attribution.AttributionRefusal):
            attribution.alignment_from_record({
                "rotation_quarter_turns": 0, "translation": [0, 0, 0],
                "transform": attribution.ALIGNMENT_TRANSFORM, forbidden: 1})
    with pytest.raises(attribution.AttributionRefusal):
        attribution.alignment_from_record({
            "rotation_quarter_turns": 0, "translation": [0, 0, 0],
            "transform": "candidate rotated, then scaled to fit"})


def test_state_properties_rotate_with_the_permitted_quarter_turn():
    aligned = attribution.align_candidate(
        {(0, 0, 0): STAIRS_NORTH}, _alignment(1))
    assert dict(aligned[(0, 0, 0)].properties) == {"facing": "east"}
    identity = attribution.align_candidate({(0, 0, 0): STAIRS_NORTH}, _alignment(0))
    assert identity[(0, 0, 0)] is STAIRS_NORTH


def test_align_palette_states_rotates_states_only_and_ignores_translation():
    """A coordinate-free palette receives the state rotation and no translation."""
    identity = attribution.align_palette_states(
        (STAIRS_NORTH, STONE), _alignment(0, (5, -3, 2)))
    assert identity == frozenset({STAIRS_NORTH, STONE})
    turned = attribution.align_palette_states(
        (STAIRS_NORTH, STONE), _alignment(1, (5, -3, 2)))
    # Only the directional state moves; translation has no coordinate to act on.
    assert turned == frozenset({STAIRS_EAST, STONE})
    # The result is deterministic and independent of the palette's iteration order.
    assert turned == attribution.align_palette_states(
        [STONE, STAIRS_NORTH], _alignment(1, (5, -3, 2)))
    with pytest.raises(attribution.AttributionRefusal):
        attribution.align_palette_states(
            (STAIRS_NORTH,), {"rotation_quarter_turns": 1, "translation": [0, 0, 0],
                              "transform": attribution.ALIGNMENT_TRANSFORM,
                              "scale": 2.0})


def test_palette_coverage_helper_aligns_the_run_native_palette():
    """The standalone ``V inter palette`` helper rotates a directional palette."""
    reference = {(0, 0, 0): STAIRS_EAST, (1, 0, 0): STONE}
    covered, total = attribution.palette_coverage(
        reference=reference, visible_reference=_visible_reference(reference),
        alignment=_alignment(1, (9, 9, 9)), palette_states=(STAIRS_NORTH, STONE))
    assert (covered, total) == (2, 2)


def test_exact_state_over_palette_is_a_separate_conditional_row_in_the_same_universe():
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    score = _score(reference, dict(reference), palette=(STONE, DIRT),
                   space=attribution.MetricSpace.EXACT_STATE,
                   checkpoint="palette_label")
    rows = attribution.exact_state_metric_rows(score, kind="label")
    primary, conditional = rows
    assert primary.metric == "exact_state_recovery"
    assert primary.denominator_count == 3 and primary.value == 1.0
    assert primary.universe == attribution.REFERENCE_CELLS_UNIVERSE
    assert conditional.metric == "exact_state_recovery_in_palette"
    assert conditional.denominator_count == 2
    assert conditional.universe == primary.universe
    # The conditional numerator is ``E inter V_P``, never the whole ``E`` set: the
    # out-of-palette GOLD exact cell is excluded, so the conditional rate is 2/2
    # (1.0) over its own |V_P| ceiling - not a 3/2 rate above that ceiling.
    assert conditional.numerator == 2
    assert conditional.value == 1.0
    # A truth state outside the palette cannot be reachable at any K.
    assert score.palette_coverage == 2
    assert score.conditional_denominator == 2
    assert score.palette_cells == frozenset({(0, 0, 0), (1, 0, 0)})


# ---------------------------------------------------------------------------
# Candidate availability
# ---------------------------------------------------------------------------

def test_candidate_availability_counts_a_state_that_loses_the_argmin():
    reference = {(0, 0, 0): GOLD, (1, 0, 0): STONE}
    # A run can only rank states its own palette holds: DIRT is a legitimate
    # in-palette distractor here, so the ranked lists stay inside the vocabulary.
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE, GOLD, DIRT), checkpoint="candidate_k")
    ranked = {(0, 0, 0): (STONE, GOLD), (1, 0, 0): (STONE, DIRT)}
    rows = attribution.candidate_metric_rows(
        score, depths=(1, 2), reachable_by_coordinate=ranked)
    by_metric = {row.metric: row for row in rows}
    # The truth state at (0,0,0) is the runner-up, so K=1 misses it and K=2 finds it.
    assert by_metric["candidate_available_k1"].numerator == 1
    assert by_metric["candidate_available_k2"].numerator == 2
    assert by_metric["candidate_available_k2"].denominator_count == 2
    # Both visible truths are in-palette, so the ceiling is the palette coverage.
    assert score.palette_coverage == 2
    for k in (1, 2):
        assert by_metric[f"candidate_available_k{k}"].numerator \
            <= score.palette_coverage


def test_candidate_availability_charges_a_beyond_palette_state_to_the_palette():
    reference = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE,), checkpoint="candidate_k")
    rows = attribution.candidate_metric_rows(
        score, depths=(8,), reachable_by_coordinate={
            (0, 0, 0): (STONE,), (1, 0, 0): (STONE,)})
    assert rows[0].numerator == 1
    assert score.palette_coverage == 1
    assert rows[0].denominator_count == 2
    # The out-of-palette GOLD truth cell is excluded from the numerator even at
    # K=8 (its ranked list is empty of GOLD): the candidate ceiling can never
    # exceed ``palette_coverage``, while the primary |V| denominator is unchanged.
    assert score.palette_cells == frozenset({(0, 0, 0)})
    assert rows[0].numerator <= score.palette_coverage
    assert rows[0].denominator_count == score.visible_cells


def test_candidate_ceiling_refuses_an_out_of_palette_ranked_candidate():
    """An out-of-palette ranked state is a producer contradiction, never a count.

    Visible truth is ``GOLD`` while the run palette is ``{STONE}``.  A malformed
    (or foreign) ranked list that offers ``GOLD`` would let the old
    ``truth in candidates`` test count the cell and overstate candidate reach, so
    the row must refuse atomically instead of filtering ``GOLD`` out before the
    top-``K`` cut.
    """
    reference = {(0, 0, 0): GOLD}
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE,), checkpoint="candidate_k")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.candidate_metric_rows(
            score, depths=(1, 8),
            reachable_by_coordinate={(0, 0, 0): (GOLD,)})
    # The contradiction is detected from the full ranked list, before any top-K
    # slice: an out-of-palette entry below the cut still refuses.
    with pytest.raises(attribution.AttributionRefusal):
        attribution.candidate_metric_rows(
            score, depths=(1,),
            reachable_by_coordinate={(0, 0, 0): (STONE, GOLD)})
    # The same truth cell with an all-palette ranked list is a normal ceiling row:
    # GOLD is not in V_P, so it is charged to the palette, not to a contradiction.
    rows = attribution.candidate_metric_rows(
        score, depths=(1,), reachable_by_coordinate={(0, 0, 0): (STONE,)})
    assert rows[0].numerator == 0
    assert rows[0].denominator_count == 1
    assert score.palette_coverage == 0
    assert score.palette_cells == frozenset()


def test_candidate_ceiling_refuses_missing_palette_membership_evidence():
    """A C score without aligned palette membership is typed unknown, not empty."""
    reference = {(0, 0, 0): STONE}
    no_palette = _score(reference, dict(reference),
                        space=attribution.MetricSpace.CANDIDATE,
                        palette=None, checkpoint="candidate_k")
    assert no_palette.aligned_palette is None
    assert no_palette.palette_cells is None
    assert no_palette.palette_coverage is None
    with pytest.raises(attribution.AttributionRefusal):
        attribution.candidate_metric_rows(
            no_palette, depths=(1,),
            reachable_by_coordinate={(0, 0, 0): (STONE,)})
    with pytest.raises(attribution.AttributionRefusal):
        attribution.candidate_metric_rows(
            no_palette, depths=(1,), reachable_by_coordinate=None)
    # A C score built from a deliberately empty palette carries membership evidence
    # (``V_P = {}``) and refuses any ranked candidate as a contradiction.
    empty_palette = _score(reference, dict(reference),
                           space=attribution.MetricSpace.CANDIDATE,
                           palette=(), checkpoint="candidate_k")
    assert empty_palette.palette_cells == frozenset()
    with pytest.raises(attribution.AttributionRefusal):
        attribution.candidate_metric_rows(
            empty_palette, depths=(1,),
            reachable_by_coordinate={(0, 0, 0): (STONE,)})


def test_cell_score_refuses_a_forged_vp_membership_set():
    """A carried ``V_P`` is recomputed, never trusted as a subset claim.

    The prior invariant only required ``palette_cells`` to be a *subset* of the
    visible truth cells, so a manually constructed ``CellScore`` could carry an
    incomplete/superset membership set and produce incorrect C ceiling and
    conditional-E rows.  ``__post_init__`` now recomputes
    ``{v : truth(v) in aligned_palette}`` from ``coordinate_truth`` and refuses
    every mismatch, with the aggregate counts kept consistent so only the
    exact-membership check can fire.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE, DIRT), checkpoint="candidate_k")
    assert score.palette_cells == frozenset({(0, 0, 0), (1, 0, 0)})
    # Incomplete: drop a genuinely covered cell.
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(score, palette_cells=frozenset({(0, 0, 0)}),
                            palette_coverage=1, conditional_denominator=1)
    # Superset: add a visible cell whose truth is outside the aligned palette.
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(
            score, palette_cells=frozenset({(0, 0, 0), (1, 0, 0), (2, 0, 0)}),
            palette_coverage=3, conditional_denominator=3)
    # A cell that is not in the visible truth context at all.
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(
            score, palette_cells=frozenset({(0, 0, 0), (9, 9, 9)}),
            palette_coverage=2, conditional_denominator=2)
    # Same-size swap: replace a covered cell with the out-of-palette GOLD cell,
    # keeping both aggregate counts identical so only exact membership can fire.
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(
            score, palette_cells=frozenset({(0, 0, 0), (2, 0, 0)}),
            palette_coverage=2, conditional_denominator=2)
    # An exact recomputation of the same set is still accepted.
    assert dataclasses.replace(
        score, palette_cells=frozenset({(0, 0, 0), (1, 0, 0)}),
        palette_coverage=2, conditional_denominator=2).palette_cells \
        == frozenset({(0, 0, 0), (1, 0, 0)})


def test_cell_score_refuses_palette_aggregates_without_complete_context():
    """A palette aggregate alone cannot reach the conditional-E fallback.

    The prior ``exact_state_over_palette`` used the whole ``len(exact)`` whenever
    ``conditional_denominator`` was present but the membership fields were absent,
    so an aggregate-only record could report ``3/2``.  The invariant now refuses
    any palette aggregate without the complete context (aligned vocabulary, exact
    ``V_P`` and ``coordinate_truth``), making that fallback state unconstructible.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.EXACT_STATE,
                   palette=(STONE, DIRT), checkpoint="palette_label")
    assert score.conditional_denominator == 2
    # Replacement form: strip the evidence pair, keep the aggregates.
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(score, aligned_palette=None, palette_cells=None)
    # Constructor form: aggregates passed with no vocabulary/V_P at all.
    with pytest.raises(attribution.AttributionRefusal):
        attribution.CellScore(
            space=score.space, checkpoint=score.checkpoint, kind=score.kind,
            universe=score.universe, denominator=score.denominator,
            alignment=score.alignment, visible_cells=score.visible_cells,
            matched=score.matched, exact=score.exact,
            aligned_cells=score.aligned_cells, phantom_cells=score.phantom_cells,
            hidden_inferred_cells=score.hidden_inferred_cells,
            hidden_exact_cells=score.hidden_exact_cells,
            palette_coverage=2, conditional_denominator=2,
            coordinate_truth=score.coordinate_truth)
    # A valid palette score has no whole-E fallback: E=3 exact cells, |V_P|=2, so
    # the conditional row is exactly 2/2 rather than 3/2.
    conditional = attribution.exact_state_metric_rows(
        score, kind="label")[1]
    assert conditional.numerator == 2
    assert conditional.denominator_count == 2
    assert conditional.value == 1.0
    # No palette aggregate means no conditional row at all.
    no_palette = _score(reference, dict(reference),
                        space=attribution.MetricSpace.EXACT_STATE,
                        palette=None, checkpoint="palette_label")
    assert no_palette.exact_state_over_palette is None
    assert len(attribution.exact_state_metric_rows(no_palette, kind="label")) == 1


def test_cell_score_refuses_palette_evidence_without_truth_context():
    """Palette evidence without ``coordinate_truth`` fails closed."""
    reference = {(0, 0, 0): STONE}
    score = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE,), checkpoint="candidate_k")
    assert score.coordinate_truth is not None
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(score, coordinate_truth=None)
    with pytest.raises(attribution.AttributionRefusal):
        dataclasses.replace(score, coordinate_truth=[(0, 0, 0)])
    # No palette evidence needs no truth context (typed unknown, not empty).
    no_palette = _score(reference, dict(reference),
                        space=attribution.MetricSpace.CANDIDATE,
                        palette=None, checkpoint="candidate_k")
    assert dataclasses.replace(
        no_palette, coordinate_truth=None).palette_cells is None


def test_cell_score_truth_snapshot_is_read_only_and_caller_isolated():
    """The carried truth context is an immutable snapshot, not the caller's dict.

    ``coordinate_truth`` used to stay a mutable mapping after the exact ``V_P``
    validation, so ``score.coordinate_truth[cell] = other`` could silently stale
    the validated evidence.  The record now stores a read-only view over a private
    copy: item assignment/deletion is rejected and the constructor's source mapping
    is decoupled from the snapshot.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    source = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    built = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE,), checkpoint="candidate_k")
    score = dataclasses.replace(built, coordinate_truth=source)
    assert score.coordinate_truth == {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    assert score.coordinate_truth is not source
    with pytest.raises(TypeError):
        score.coordinate_truth[(0, 0, 0)] = DIRT
    with pytest.raises(TypeError):
        del score.coordinate_truth[(0, 0, 0)]
    # Mutating the constructor's source mapping after construction cannot change
    # the frozen snapshot or the validated ``V_P`` membership.
    source[(0, 0, 0)] = DIRT
    del source[(1, 0, 0)]
    assert score.coordinate_truth == {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    assert score.palette_cells == frozenset({(0, 0, 0)})
    assert score.palette_coverage == 1 and score.conditional_denominator == 1


def test_candidate_rows_stay_stable_after_source_truth_mutation():
    """C candidate reach reads the frozen truth snapshot, not the source dict."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    source = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    built = _score(reference, dict(reference),
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=(STONE,), checkpoint="candidate_k")
    score = dataclasses.replace(built, coordinate_truth=source)
    ranked = {(0, 0, 0): (STONE,), (1, 0, 0): (STONE,)}
    before = attribution.candidate_metric_rows(
        score, depths=(1,), reachable_by_coordinate=ranked)[0]
    # Rewriting the covered cell's truth out of the palette would drop it from the
    # numerator if the live source dict were read instead of the snapshot.
    source[(0, 0, 0)] = DIRT
    source[(1, 0, 0)] = STONE
    after = attribution.candidate_metric_rows(
        score, depths=(1,), reachable_by_coordinate=ranked)[0]
    assert before.numerator == 1
    assert after.numerator == 1
    assert after.denominator_count == before.denominator_count == 2
    assert score.palette_cells == frozenset({(0, 0, 0)})


def test_conditional_exact_state_stays_stable_after_source_truth_mutation():
    """``E inter V_P`` also reads the frozen snapshot, so its rate cannot drift."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    source = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    built = _score(reference, dict(reference),
                   space=attribution.MetricSpace.EXACT_STATE,
                   palette=(STONE,), checkpoint="palette_label")
    score = dataclasses.replace(built, coordinate_truth=source)
    before = score.exact_state_over_palette
    source[(0, 0, 0)] = DIRT
    source[(1, 0, 0)] = GOLD
    after = score.exact_state_over_palette
    assert score.coordinate_truth == {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    assert after == before
    assert after is not None
    assert after["numerator"] == 1 and after["denominator"] == 1
    assert score.palette_cells == frozenset({(0, 0, 0)})


# ---------------------------------------------------------------------------
# In-palette loss partition (additive evaluator-only diagnostic)
# ---------------------------------------------------------------------------

def _in_palette_partition(reference, candidate, *, palette, turns=0,
                          translation=(0, 0, 0)):
    score = _score(
        reference, candidate, space=attribution.MetricSpace.EXACT_STATE,
        palette=palette, checkpoint="palette_label", turns=turns,
        translation=translation)
    partition = attribution.in_palette_loss_partition(score)
    assert partition is not None
    return score, partition


def test_in_palette_loss_partition_splits_every_visible_in_palette_cell_once():
    """Each visible in-palette truth cell lands in exactly one of the three sets.

    The reference carries two out-of-palette ``GOLD`` truth cells so the partition
    is exercised against the full ``|V|`` denominator, including one out-of-palette
    cell whose candidate is an exact match: it is outside ``V_P`` and must appear in
    none of the three partitions.
    """
    reference = {
        (0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD,
        (3, 0, 0): STONE, (4, 0, 0): DIRT, (5, 0, 0): GOLD,
    }
    candidate = {
        (0, 0, 0): STONE,   # exact hit, in palette
        (1, 0, 0): STONE,   # covered but wrong state, in palette
        (2, 0, 0): GOLD,    # exact, yet out of palette -> never in V_P
        (4, 0, 0): DIRT,    # exact hit, in palette
        (5, 0, 0): DIRT,    # wrong state, yet out of palette
        # (3, 0, 0) is deliberately absent -> geometry missed, in palette
    }
    score, partition = _in_palette_partition(reference, candidate,
                                             palette=(STONE, DIRT))
    assert score.palette_cells == frozenset(
        {(0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)})
    assert partition.palette_cells == score.palette_cells
    assert partition.visible_cells == 6
    assert partition.geometry_missed == frozenset({(3, 0, 0)})
    assert partition.wrong_state == frozenset({(1, 0, 0)})
    assert partition.exact_hit == frozenset({(0, 0, 0), (4, 0, 0)})
    # Disjoint and exhaustive: the three sets are a partition of V_P, not a
    # numerator the aggregate conditional row could inflate above its ceiling.
    assert partition.geometry_missed & partition.wrong_state == frozenset()
    assert partition.geometry_missed & partition.exact_hit == frozenset()
    assert partition.wrong_state & partition.exact_hit == frozenset()
    assert partition.geometry_missed | partition.wrong_state | partition.exact_hit \
        == partition.palette_cells
    assert partition.candidate_covered == 3
    opportunity = partition.candidate_opportunity
    assert opportunity == {"value": 0.75, "numerator": 3, "denominator": 4,
                           "undefined_reason": None}
    exact = partition.exact_state_in_candidate_opportunity
    assert exact["numerator"] == 2 and exact["denominator"] == 3
    assert exact["value"] == pytest.approx(2 / 3)
    assert exact["undefined_reason"] is None


def test_in_palette_loss_partition_rate_is_undefined_when_opportunity_is_empty():
    """``exact / |C inter V_P|`` is typed null, never a measured zero, at 0/0."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT}
    # V_P is nonempty but no candidate covers any in-palette cell: the geometry is
    # missed everywhere and the conditional opportunity denominator is empty.
    score, partition = _in_palette_partition(reference, {},
                                             palette=(STONE, DIRT))
    assert partition.palette_cells == frozenset({(0, 0, 0), (1, 0, 0)})
    assert partition.candidate_covered == 0
    assert partition.geometry_missed == frozenset({(0, 0, 0), (1, 0, 0)})
    assert partition.wrong_state == frozenset()
    assert partition.exact_hit == frozenset()
    opportunity = partition.candidate_opportunity
    assert opportunity["value"] == 0.0
    assert opportunity["numerator"] == 0 and opportunity["denominator"] == 2
    conditional = partition.exact_state_in_candidate_opportunity
    assert conditional["value"] is None
    assert conditional["numerator"] == 0 and conditional["denominator"] == 0
    assert conditional["undefined_reason"] == "no_candidate_covered_palette_cell"
    rows = {row.metric: row for row in
            attribution.in_palette_loss_partition_rows(score)}
    assert rows["exact_state_in_candidate_opportunity"].value is None
    assert rows["exact_state_in_candidate_opportunity"].undefined_reason == \
        "no_candidate_covered_palette_cell"


def test_in_palette_loss_partition_requires_palette_and_exact_state_evidence():
    """Absent palette is typed unknown; a geometry-only score is a refusal."""
    reference = {(0, 0, 0): STONE}
    no_palette = _score(reference, dict(reference),
                        space=attribution.MetricSpace.EXACT_STATE,
                        palette=None, checkpoint="palette_label")
    assert attribution.in_palette_loss_partition(no_palette) is None
    with pytest.raises(attribution.AttributionRefusal):
        attribution.in_palette_loss_partition_rows(no_palette)
    # A geometry-only candidate has no state to classify as wrong: refusing keeps
    # every covered in-palette cell from being mislabelled as a wrong state.
    geometry = _score(reference, dict(reference),
                      space=attribution.MetricSpace.GEOMETRY,
                      palette=(STONE,), checkpoint="facade_grid")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.in_palette_loss_partition(geometry)
    # An explicitly empty palette is a measured V_P = {}, so both rates are null.
    empty = _score(reference, {},
                   space=attribution.MetricSpace.EXACT_STATE,
                   palette=(), checkpoint="palette_label")
    partition = attribution.in_palette_loss_partition(empty)
    assert partition is not None
    assert partition.palette_cells == frozenset()
    assert partition.candidate_opportunity["value"] is None
    assert partition.candidate_opportunity["undefined_reason"] == \
        "no_visible_cell_in_run_palette"
    assert partition.exact_state_in_candidate_opportunity["value"] is None


def test_in_palette_loss_partition_record_refuses_a_forged_partition():
    """A manually built partition must still be disjoint and cover V_P exactly."""
    with pytest.raises(attribution.AttributionRefusal):
        attribution.InPaletteLossPartition(
            palette_cells=frozenset({(0, 0, 0), (1, 0, 0)}), visible_cells=2,
            geometry_missed=frozenset({(0, 0, 0)}), wrong_state=frozenset(),
            exact_hit=frozenset())
    with pytest.raises(attribution.AttributionRefusal):
        attribution.InPaletteLossPartition(
            palette_cells=frozenset({(0, 0, 0)}), visible_cells=1,
            geometry_missed=frozenset({(0, 0, 0)}),
            wrong_state=frozenset({(0, 0, 0)}), exact_hit=frozenset())
    with pytest.raises(attribution.AttributionRefusal):
        attribution.InPaletteLossPartition(
            palette_cells=frozenset({(0, 0)}), visible_cells=1,
            geometry_missed=frozenset({(0, 0)}), wrong_state=frozenset(),
            exact_hit=frozenset())


def test_in_palette_loss_partition_record_is_strict_canonical_json():
    """The serialized partition is canonical JSON counts and rates, no coordinates."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    candidate = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    _, partition = _in_palette_partition(reference, candidate,
                                         palette=(STONE, DIRT))
    record = partition.to_record()
    decoded = json.loads(attribution._canonical(record))
    assert decoded["visible_cells"] == 3
    assert decoded["visible_in_palette_cells"] == 2
    assert decoded["candidate_covered_in_palette_cells"] == 2
    assert decoded["geometry_missed_cells"] == 0
    assert decoded["wrong_state_cells"] == 1
    assert decoded["exact_state_hit_cells"] == 1
    assert decoded["candidate_opportunity"]["value"] == 1.0
    assert decoded["exact_state_in_candidate_opportunity"]["value"] == 0.5
    # In-memory cell membership is evidence, not payload: no coordinate triple and
    # no raw set is serialized, and no value is a list of coordinates.
    assert "palette_cells" not in decoded
    assert "(0, 0, 0)" not in json.dumps(decoded, sort_keys=True)
    assert all(not isinstance(value, list) for value in decoded.values())


def test_in_palette_loss_partition_follows_the_permitted_nonidentity_alignment():
    """The partition reads the frozen aligned memberships, applying no second transform."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    turns, translation = 1, (2, 0, -1)
    inverse = _inverse_translation(translation, turns)
    run_cells = {
        cell: tuple(_unrot(cell, turns)[axis] + inverse[axis] for axis in range(3))
        for cell in reference}
    aligned_candidate = {(0, 0, 0): STONE, (2, 0, 0): GOLD}
    candidate = {run_cells[cell]: state for cell, state in aligned_candidate.items()}
    score, partition = _in_palette_partition(
        reference, candidate, palette=(STONE, DIRT), turns=turns,
        translation=translation)
    assert score.palette_cells == frozenset({(0, 0, 0), (1, 0, 0)})
    assert partition.geometry_missed == frozenset({(1, 0, 0)})
    assert partition.wrong_state == frozenset()
    assert partition.exact_hit == frozenset({(0, 0, 0)})
    assert partition.candidate_opportunity == {
        "value": 0.5, "numerator": 1, "denominator": 2, "undefined_reason": None}
    assert partition.exact_state_in_candidate_opportunity["value"] == 1.0


def test_in_palette_loss_partition_rows_are_additive_and_do_not_change_existing_rows():
    """Requesting the partition rows leaves every existing row and score unchanged."""
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    candidate = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    score, _ = _in_palette_partition(reference, candidate, palette=(STONE, DIRT))
    c_score = _score(reference, candidate,
                     space=attribution.MetricSpace.CANDIDATE,
                     palette=(STONE, DIRT, GOLD), checkpoint="candidate_k")
    ranked = {(0, 0, 0): (STONE,), (1, 0, 0): (GOLD,)}
    before_exact = [row.to_record() for row in
                    attribution.exact_state_metric_rows(score, kind="label")]
    before_candidate = [row.to_record() for row in attribution.candidate_metric_rows(
        c_score, depths=(1,), reachable_by_coordinate=ranked)]
    before_score = score.to_record()
    before_c_score = c_score.to_record()
    rows = attribution.in_palette_loss_partition_rows(score)
    by_metric = {row.metric: row for row in rows}
    assert set(by_metric) == {
        "in_palette_geometry_missed", "in_palette_wrong_state",
        "in_palette_exact_state_hit", "candidate_opportunity_in_palette",
        "exact_state_in_candidate_opportunity"}
    assert by_metric["in_palette_geometry_missed"].numerator == 0
    assert by_metric["in_palette_geometry_missed"].denominator_count == 2
    assert by_metric["in_palette_wrong_state"].numerator == 1
    assert by_metric["in_palette_exact_state_hit"].numerator == 1
    assert by_metric["candidate_opportunity_in_palette"].value == 1.0
    assert by_metric["exact_state_in_candidate_opportunity"].value == 0.5
    assert by_metric["exact_state_in_candidate_opportunity"].is_accuracy
    for row in rows:
        assert row.space is attribution.MetricSpace.EXACT_STATE
        assert row.universe == attribution.REFERENCE_CELLS_UNIVERSE
        assert row.reached and row.metric_value is not None
        assert attribution._canonical(row.to_record())
    # No existing builder was touched: the additive rows are absent from them and
    # the exact/candidate rows and both score records serialize identically.
    assert "in_palette_geometry_missed" not in {
        row.metric for row in attribution.exact_state_metric_rows(score, kind="label")}
    assert [row.to_record() for row in
            attribution.exact_state_metric_rows(score, kind="label")] == before_exact
    assert [row.to_record() for row in attribution.candidate_metric_rows(
        c_score, depths=(1,), reachable_by_coordinate=ranked)] == before_candidate
    assert score.to_record() == before_score
    assert c_score.to_record() == before_c_score
    assert score.alignment == c_score.alignment


# ---------------------------------------------------------------------------
# Transitions: non-monotone, in-universe, disjoint loss partition
# ---------------------------------------------------------------------------

def test_exact_state_transition_reconciles_and_partitions_loss_disjointly():
    reference = {(0, 0, 0): STONE, (1, 0, 0): STONE, (2, 0, 0): STONE,
                 (3, 0, 0): STONE}
    earlier = _score(reference, dict(reference),
                     space=attribution.MetricSpace.EXACT_STATE,
                     checkpoint="palette_label")
    # The later checkpoint keeps (0,0,0) and (1,0,0), loses (2,0,0) to a wrong
    # label and drops (3,0,0) from geometry entirely.
    later = _score(reference, {(0, 0, 0): STONE, (1, 0, 0): STONE,
                               (2, 0, 0): DIRT},
                   space=attribution.MetricSpace.EXACT_STATE,
                   checkpoint="terminal")
    transition = attribution.cell_transition(earlier, later)
    assert transition.previous == 4
    assert transition.lost == 2
    assert transition.gained == 0
    assert transition.geometry_lost == 1
    assert transition.state_lost == 1
    assert transition.geometry_lost + transition.state_lost == transition.lost
    assert transition.net == -2
    assert transition.to_record()["identity"] == "next = previous + gained - lost"


def test_a_transition_never_spans_two_universes_or_the_camera_diagnostic():
    reference = {(0, 0, 0): STONE}
    geometry = _score(reference, dict(reference))
    exact = _score(reference, dict(reference),
                   space=attribution.MetricSpace.EXACT_STATE)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.cell_transition(geometry, exact)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionTransition(
            universe=attribution.FRAME_PIXELS_UNIVERSE,
            previous_checkpoint="a", next_checkpoint="b",
            space=attribution.MetricSpace.GEOMETRY,
            previous=1, gained=0, lost=1, net=-1)


def test_non_monotone_gain_is_reported_and_geometry_partition_is_rejected_off_space():
    reference = {(0, 0, 0): STONE, (1, 0, 0): STONE}
    earlier = _score(reference, {(0, 0, 0): DIRT},
                     space=attribution.MetricSpace.EXACT_STATE,
                     checkpoint="palette_label")
    later = _score(reference, dict(reference),
                   space=attribution.MetricSpace.EXACT_STATE,
                   checkpoint="terminal")
    transition = attribution.cell_transition(earlier, later)
    assert transition.gained == 2 and transition.lost == 0
    assert transition.net == 2
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionTransition(
            universe=attribution.REFERENCE_CELLS_UNIVERSE,
            previous_checkpoint="a", next_checkpoint="b",
            space=attribution.MetricSpace.GEOMETRY,
            previous=1, gained=0, lost=1, net=-1, geometry_lost=1, state_lost=0)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionTransition(
            universe=attribution.REFERENCE_CELLS_UNIVERSE,
            previous_checkpoint="a", next_checkpoint="b",
            space=attribution.MetricSpace.EXACT_STATE,
            previous=2, gained=0, lost=1, net=-1,
            geometry_lost=1, state_lost=1)  # partition does not sum to lost


# ---------------------------------------------------------------------------
# Typed unavailable rows and fail-closed behaviour
# ---------------------------------------------------------------------------

def test_an_unreached_row_is_typed_null_and_never_a_zero():
    row = attribution._unreached_row(
        checkpoint="facade_grid", metric="geometry_recall",
        space=attribution.MetricSpace.GEOMETRY, reason="no_visible_blocks")
    assert row.reached is False
    assert row.metric_value is None
    assert row.ceiling is None
    assert row.conditional_denominator is None
    assert row.typed_reason == "no_visible_blocks"
    # A row built without production-checkpoint context states no reach; it is
    # never inferred from the metric's absence.
    assert row.checkpoint_reached is None
    with pytest.raises(attribution.AttributionRefusal):
        attribution._unreached_row(
            checkpoint="facade_grid", metric="geometry_recall",
            space=attribution.MetricSpace.GEOMETRY, reason="two words")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionRow(
            checkpoint="facade_grid", metric="geometry_recall",
            space=attribution.MetricSpace.GEOMETRY, kind="geometry",
            universe=attribution.REFERENCE_CELLS_UNIVERSE,
            denominator="V", reached=False, typed_reason="absent",
            metric_value={"value": 0.0, "numerator": 0, "denominator": 4,
                          "undefined_reason": None})


def test_attribution_unavailable_is_typed_and_contributes_no_zero():
    section = attribution.attribution_unavailable(
        "no_admitted_frames", stage="lattice", detail="mask_unavailable")
    assert section["available"] is False
    assert section["metric"] is None
    with pytest.raises(attribution.AttributionRefusal):
        attribution.attribution_unavailable("")


def test_completion_and_refusal_rates_are_policy_rates_never_accuracy():
    rates = attribution.completion_and_refusal_rates(
        planned=5, reached=3, terminal_stage="geometry", frontier_stop="output")
    assert rates["completion_rate"]["numerator"] == 3
    assert rates["completion_rate"]["denominator"] == 5
    assert rates["refusal_rate"]["numerator"] == 2
    assert "never accuracy" in rates["accounting"]
    undefined = attribution.completion_and_refusal_rates(
        planned=0, reached=0, terminal_stage="", frontier_stop=None)
    assert undefined["completion_rate"]["value"] is None


# ---------------------------------------------------------------------------
# Route A builder
# ---------------------------------------------------------------------------

def _route_a_frontier(*, with_facade=True, with_export=True, camera=True):
    image = type("Image", (), {"shape": (4, 5, 3)})()
    facade = _Volume({(0, 0, 0): STONE}) if with_facade else None
    decoded = _Volume({(0, 0, 0): STONE}) if with_export else None
    palette = (_frame(),)
    camera_model = None
    if camera:
        camera_model = SimpleNamespace(
            position=(0.0, 0.0, 0.0), yaw_degrees=0.0, pitch_degrees=0.0,
            roll_degrees=0.0, fov_degrees=70.0)
    return route_a_frontier(
        admission=image, camera=camera_model, facade=facade, palette=palette,
        decoded_export=decoded, camera_solved=camera).frontier


class _Volume:
    def __init__(self, blocks):
        self.blocks = dict(blocks)

    def to_dict(self):
        return json.dumps({"blocks": [[list(key), str(value.name)]
                                      for key, value in sorted(self.blocks.items())]},
                          sort_keys=True, separators=(",", ":"))


def _frame():
    return SimpleNamespace(state=STONE, rgb=(125, 125, 125))


def _route_a_attribution(*, frontier=None, facade=_SENTINEL, terminal=_SENTINEL):
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT}
    return attribution.route_a_attribution(
        frontier=frontier or _route_a_frontier(),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(),
        facade=dict(reference) if facade is _SENTINEL else facade,
        palette=(STONE, DIRT),
        palette_terminal=dict(reference) if terminal is _SENTINEL else terminal)


def test_route_a_attributes_its_own_geometry_palette_and_terminal_rows():
    record = _route_a_attribution()
    sections = record.section
    assert sections[attribution.MetricSpace.GEOMETRY].rows
    assert sections[attribution.MetricSpace.CANDIDATE].rows
    assert sections[attribution.MetricSpace.EXACT_STATE].rows
    # O and P are never borrowed from Route A's cell counts.
    assert sections[attribution.MetricSpace.OWNERSHIP].rows == ()
    assert sections[attribution.MetricSpace.PAIR_CORRESPONDENCE].rows == ()
    checkpoints = {row.checkpoint for row in sections[attribution.MetricSpace.GEOMETRY].rows}
    assert checkpoints == {"facade_grid"}
    # Every scored Route A row belongs to a checkpoint the frontier reached.
    scored_rows = [row for section in record.sections for row in section.rows]
    assert scored_rows and all(row.checkpoint_reached is True for row in scored_rows)
    assert record.frontier_stop is None
    assert record.route.value == "route_a"
    # The one comparable Route A transition joins palette labels -> decoded export.
    transitions = sections[attribution.MetricSpace.EXACT_STATE].transitions
    assert [(item.previous_checkpoint, item.next_checkpoint) for item in transitions] == \
        [("palette_label", "terminal")]


def test_route_a_without_a_facade_is_typed_unavailable_not_zero():
    record = _route_a_attribution(
        frontier=_route_a_frontier(with_facade=False, with_export=False),
        facade=None, terminal=None)
    rows = record.section[attribution.MetricSpace.GEOMETRY].rows
    assert rows and all(row.reached is False for row in rows)
    assert all(row.metric_value is None for row in rows)
    assert all(row.typed_reason for row in rows)
    # The facade checkpoint itself was not reached here, so reach is false, not null.
    assert all(row.checkpoint_reached is False for row in rows)
    assert record.frontier_stop is not None


def test_route_a_carries_the_unknown_scale_caveat_not_an_unavailable_geometry():
    record = _route_a_attribution()
    assert any("scale" in caveat for caveat in record.caveats)
    geometry_rows = record.section[attribution.MetricSpace.GEOMETRY].rows
    assert geometry_rows and all(row.reached for row in geometry_rows)


# ---------------------------------------------------------------------------
# Route A opt-in in-palette loss partition exposure
# ---------------------------------------------------------------------------

PARTITION_METRICS = frozenset({
    "in_palette_geometry_missed", "in_palette_wrong_state",
    "in_palette_exact_state_hit", "candidate_opportunity_in_palette",
    "exact_state_in_candidate_opportunity"})


def _route_a_partition_record(*, include=False, facade=_SENTINEL, palette=_SENTINEL):
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): GOLD}
    default_facade = {(0, 0, 0): STONE, (1, 0, 0): GOLD}
    return attribution.route_a_attribution(
        frontier=_route_a_frontier(),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(),
        facade=default_facade if facade is _SENTINEL else facade,
        palette=(STONE, DIRT, GOLD) if palette is _SENTINEL else palette,
        palette_terminal=None, include_palette_loss_partition=include)


def _section_rows(record):
    return [row.to_record() for section in record.sections for row in section.rows]


def test_route_a_partition_option_default_off_is_byte_for_byte_unchanged():
    """Absent (or explicitly false), the option adds nothing and moves nothing."""
    default = _route_a_partition_record()
    explicit_off = _route_a_partition_record(include=False)
    assert attribution.attribution_digest(default) == \
        attribution.attribution_digest(explicit_off)
    assert attribution._canonical(default.to_record()) == \
        attribution._canonical(explicit_off.to_record())
    metrics = {row["metric"] for row in _section_rows(default)}
    assert metrics & PARTITION_METRICS == set()
    # The partition stays behind its own explicit opt-in: no default section row
    # and no default builder call carries it.
    exact_rows = [row.to_record() for row in attribution.cell_metric_rows(
        _score({(0, 0, 0): STONE}, {(0, 0, 0): STONE},
               space=attribution.MetricSpace.EXACT_STATE, palette=(STONE,),
               checkpoint="palette_label"), kind="label")]
    assert {row["metric"] for row in exact_rows} & PARTITION_METRICS == set()


def test_route_a_partition_option_appends_exactly_five_palette_label_rows():
    """Opt-in appends the reviewed five rows and leaves every default row in place."""
    default = _route_a_partition_record()
    opt_in = _route_a_partition_record(include=True)
    default_rows = _section_rows(default)
    opt_rows = _section_rows(opt_in)
    added = [row for row in opt_rows if row["metric"] in PARTITION_METRICS]
    assert len(added) == 5
    # Every pre-existing row survives in the same relative order: the opt-in is
    # purely additive and touches no existing denominator or payload.
    assert [row for row in opt_rows if row["metric"] not in PARTITION_METRICS] == \
        default_rows
    assert all(row["checkpoint"] == "palette_label" for row in added)
    assert all(row["space"] == "E" for row in added)
    assert all(row["universe"] == attribution.REFERENCE_CELLS_UNIVERSE
               for row in added)
    assert all(row["reached"] is True for row in added)
    indexes = [index for index, row in enumerate(opt_rows)
               if row["metric"] in PARTITION_METRICS]
    assert indexes == list(range(indexes[0], indexes[0] + 5))
    by_metric = {row["metric"]: row for row in added}
    # reference V = 3, V_P = 3 (all truths in palette); (0,0,0) exact hit,
    # (1,0,0) covered-wrong, (2,0,0) geometry missed.
    assert by_metric["in_palette_geometry_missed"]["metric_value"]["numerator"] == 1
    assert by_metric["in_palette_wrong_state"]["metric_value"]["numerator"] == 1
    assert by_metric["in_palette_exact_state_hit"]["metric_value"]["numerator"] == 1
    assert by_metric["candidate_opportunity_in_palette"]["metric_value"]["value"] == \
        pytest.approx(2 / 3)
    assert by_metric["exact_state_in_candidate_opportunity"]["metric_value"]["value"] == \
        0.5
    assert by_metric["exact_state_in_candidate_opportunity"]["is_accuracy"] is True
    assert attribution.attribution_digest(opt_in) != \
        attribution.attribution_digest(default)


def test_route_a_partition_option_refuses_without_palette_evidence():
    """A request with no reached palette_label palette evidence is a typed refusal."""
    reference = {(0, 0, 0): STONE}
    visible = _visible_reference(reference)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.route_a_attribution(
            frontier=_route_a_frontier(), visible_reference=visible,
            reference=reference, alignment=_alignment(), facade=dict(reference),
            palette=None, palette_terminal=None,
            include_palette_loss_partition=True)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.route_a_attribution(
            frontier=_route_a_frontier(with_facade=False, with_export=False),
            visible_reference=visible, reference=reference, alignment=_alignment(),
            facade=None, palette=(STONE,), palette_terminal=None,
            include_palette_loss_partition=True)
    with pytest.raises(attribution.AttributionRefusal):
        attribution.route_a_attribution(
            frontier=_route_a_frontier(), visible_reference=visible,
            reference=reference, alignment=_alignment(), facade=dict(reference),
            palette=(STONE,), palette_terminal=None,
            include_palette_loss_partition="yes")


def test_build_attribution_rejects_the_partition_option_on_route_b():
    """Route B has no palette_label checkpoint, so the option is a typed refusal."""
    reference = {(0, 0, 0): STONE}
    with pytest.raises(attribution.AttributionRefusal) as error:
        attribution.build_attribution(
            frontier=_synthetic_route_b_frontier(),
            visible_reference=_visible_reference(reference), reference=reference,
            alignment=_alignment(), include_palette_loss_partition=True)
    assert attribution.PALETTE_LOSS_PARTITION_ROUTE_B_REASON in str(error.value)


# ---------------------------------------------------------------------------
# Route B builder
# ---------------------------------------------------------------------------

def _synthetic_route_b_frontier(*, geometry_stop=True, full_tail=False):
    entry = SimpleNamespace(
        entry_sha256="a" * 64, status="ok", reason="",
        session=SimpleNamespace(session_sha256="b" * 64, frames=(1, 2, 3)))
    lattice = SimpleNamespace(
        status="assembled", reason="", detail="",
        result_sha256="c" * 64, session_sha256="b" * 64)
    geometry = SimpleNamespace(
        status="unavailable" if geometry_stop else "assembled", reason="",
        detail="plane_integer_residual", result_sha256="d" * 64,
        lattice_sha256="c" * 64)
    if full_tail:
        output = SimpleNamespace(
            status="promoted", reason="", detail="",
            result_sha256="e" * 64, geometry_sha256="d" * 64)
        result = SimpleNamespace(
            schema="mc-photogrammetry-image-derived-production", producer="p",
            stage="output", status="promoted", reason="", detail="",
            entry_sha256="a" * 64, lattice_sha256="c" * 64,
            geometry_sha256="d" * 64, output_sha256="e" * 64,
            result_sha256="e" * 64)
        return route_b_frontier(entry=entry, lattice=lattice, geometry=geometry,
                                output=output, result=result)
    return route_b_frontier(entry=entry, lattice=lattice, geometry=geometry)


def test_route_b_genuine_stop_scores_placed_and_nulls_after_the_first_absent_record():
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(),
        placement={(0, 0, 0): STONE, (1, 0, 0): DIRT},
        palette=(STONE, DIRT))
    rows = [row for section in record.sections for row in section.rows]
    placed = [row for row in rows if row.checkpoint == "placed"]
    assert placed and all(row.reached for row in placed)
    # Every checkpoint after the absent output record is typed unavailable/null.
    later = [row for row in rows
             if row.checkpoint in ("volume_cell", "raw_winner", "promoted", "terminal",
                                   "candidate_k")]
    assert later and all(row.reached is False for row in later)
    assert all(row.metric_value is None for row in later)
    assert all(row.typed_reason for row in later)
    # The frontier stop is the first absent record, reported separately.
    assert record.frontier_stop is not None
    assert record.frontier_stop.stage in ("output", "candidate_k")


def test_route_b_reports_ownership_as_explicitly_unavailable_not_as_a_cell_count():
    reference = {(0, 0, 0): STONE}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,))
    assert record.section[attribution.MetricSpace.OWNERSHIP].rows == ()
    assert "ownership" in record.section[attribution.MetricSpace.OWNERSHIP].notes


def _pair_totals(*, truth_pairs, truth_pairs_returned, returned, positive):
    return {
        "pairs": 3, "returned": returned, "positive": positive,
        "truth_pairs": truth_pairs, "truth_pairs_returned": truth_pairs_returned,
    }


_ACCEPTED_MATCH_DIGEST = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


def _bound_entry_evidence(*, truth_pairs, truth_pairs_returned, returned, positive):
    """A same-matcher, digest-equal aggregate: the only scoreable entry P kind."""
    return attribution.EntryPairEvidence(
        binding=attribution.EntryMatchBinding(
            source=attribution.JOINT_MATCHER_SOURCE,
            matches_sha256=_ACCEPTED_MATCH_DIGEST,
            accepted_matches_sha256=_ACCEPTED_MATCH_DIGEST),
        totals=_pair_totals(truth_pairs=truth_pairs,
                            truth_pairs_returned=truth_pairs_returned,
                            returned=returned, positive=positive))


def _control_entry_evidence():
    """The evaluator-only alternate matcher aggregate: a control, never achieved P."""
    return attribution.EntryPairEvidence(
        binding=attribution.EntryMatchBinding(
            source="evaluation_pair_correspondence.evaluate_scene",
            matches_sha256="a" * 64,
            accepted_matches_sha256="a" * 64),
        totals=_pair_totals(truth_pairs=232, truth_pairs_returned=0,
                            returned=33, positive=0))


def test_route_b_entry_is_measured_only_from_a_digest_bound_joint_match_set():
    """The reachable entry/B0 checkpoint is scored only from the same joint matcher.

    The row is ``truth_pairs_returned / truth_pairs`` in the frozen pixel-pair
    universe; it is never divided by ``|V|`` and never substituted by entry
    connectivity.  The accepted precision row is reported alongside it.
    """
    reference = {(0, 0, 0): STONE}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,),
        entry_pair_evidence=_bound_entry_evidence(
            truth_pairs=8, truth_pairs_returned=6, returned=10, positive=7))
    section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
    assert section.universe == attribution.PIXEL_PAIRS_UNIVERSE
    rows = {row.metric: row for row in section.rows}
    assert set(rows) == {"truth_pair_recall", "positive_precision"}
    recall = rows["truth_pair_recall"]
    assert recall.checkpoint == "entry"
    assert recall.reached is True and recall.numerator == 6 and recall.denominator_count == 8
    assert recall.value == 6 / 8
    assert recall.universe == attribution.PIXEL_PAIRS_UNIVERSE
    assert recall.checkpoint_reached is True
    assert recall.to_record()["checkpoint_reached"] is True
    assert "|V|" not in recall.denominator
    precision = rows["positive_precision"]
    assert precision.numerator == 7 and precision.denominator_count == 10
    assert precision.checkpoint_reached is True
    # P starts no cell transition and is not mixed into any other space.
    assert section.transitions == ()
    assert all(row.space is attribution.MetricSpace.PAIR_CORRESPONDENCE
               for row in section.rows)


def test_route_b_entry_is_typed_unavailable_without_a_digest_bound_joint_set():
    reference = {(0, 0, 0): STONE}
    # The default typed reason is the digest-unbound contract reason; an explicit
    # reason may name a more specific missing seam.
    defaulted = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,))
    explicit = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,),
        entry_pair_unavailable_reason=attribution.ENTRY_MATCH_UNBOUND_REASON)
    for record in (defaulted, explicit):
        section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
        assert len(section.rows) == 1
        row = section.rows[0]
        assert row.checkpoint == "entry"
        assert row.reached is False
        assert row.metric_value is None
        assert row.typed_reason == attribution.ENTRY_MATCH_UNBOUND_REASON
        assert row.universe == attribution.PIXEL_PAIRS_UNIVERSE
        # This frontier reaches entry, so metric unavailability is reported beside
        # checkpoint reach rather than as an unreached checkpoint.
        assert row.checkpoint_reached is True


def test_route_b_reached_but_unscorable_entry_keeps_reach_true_and_metric_null():
    """The reached entry/B0 checkpoint is serialized apart from its null metric.

    The corrected frozen Route B run demonstrably reaches entry/B0 while retaining
    no digest-bound joint match set, so the P row must serialize
    ``checkpoint_reached: true`` beside ``reached: false``, ``metric_value: null``
    and the typed reason: a missing metric is never read as the checkpoint not
    having run, and a reached checkpoint never implies a measured metric.
    """
    reference = {(0, 0, 0): STONE}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,))
    section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.checkpoint == "entry"
    assert row.checkpoint_reached is True
    assert row.reached is False
    assert row.metric_value is None
    assert row.ceiling is None and row.conditional_denominator is None
    assert row.typed_reason == attribution.ENTRY_MATCH_UNBOUND_REASON
    assert row.is_accuracy is False
    serialized = row.to_record()
    assert serialized["checkpoint_reached"] is True
    assert serialized["reached"] is False
    assert serialized["metric_value"] is None
    assert serialized["typed_reason"] == attribution.ENTRY_MATCH_UNBOUND_REASON


def test_route_b_unreached_entry_serializes_checkpoint_reach_false():
    """An entry the run never reached is reach ``false``, not an unscorable metric."""
    reference = {(0, 0, 0): STONE}
    frontier = route_b_frontier(entry=None, blocked_reason="entry_refused")
    record = attribution.route_b_attribution(
        frontier=frontier, visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment())
    section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.checkpoint == "entry"
    assert row.checkpoint_reached is False
    assert row.reached is False
    assert row.metric_value is None
    assert row.typed_reason == "entry_unreached"
    assert row.to_record()["checkpoint_reached"] is False


def test_route_b_unreached_entry_never_emits_a_numeric_p_metric_even_when_bound():
    """A digest-bound aggregate cannot score a checkpoint A1 never reached.

    ``bound`` is necessary but not sufficient: the accepted frontier's entry reach
    is checked first, so an entry-absent frontier fails closed to
    ``entry_unreached`` with ``checkpoint_reached: false`` and emits no numeric P
    metric.  ``pair_metric_rows`` refuses that same contradiction when called
    directly, so a bound aggregate can never be attached to an unreached entry.
    """
    reference = {(0, 0, 0): STONE}
    frontier = route_b_frontier(entry=None, blocked_reason="entry_refused")
    record = attribution.route_b_attribution(
        frontier=frontier, visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        entry_pair_evidence=_bound_entry_evidence(
            truth_pairs=8, truth_pairs_returned=6, returned=10, positive=7))
    section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.checkpoint == "entry"
    assert row.checkpoint_reached is False
    assert row.reached is False
    assert row.metric_value is None
    assert row.numerator is None and row.denominator_count is None and row.value is None
    assert row.typed_reason == "entry_unreached"
    assert row.is_accuracy is False
    serialized = row.to_record()
    assert serialized["metric_value"] is None
    assert serialized["checkpoint_reached"] is False
    assert serialized["reached"] is False
    with pytest.raises(attribution.AttributionRefusal):
        attribution.pair_metric_rows(
            checkpoint="entry",
            totals=_pair_totals(truth_pairs=8, truth_pairs_returned=6,
                                returned=10, positive=7),
            binding=attribution.EntryMatchBinding(
                source=attribution.JOINT_MATCHER_SOURCE,
                matches_sha256=_ACCEPTED_MATCH_DIGEST,
                accepted_matches_sha256=_ACCEPTED_MATCH_DIGEST),
            checkpoint_reached=False)


def test_route_b_promotion_and_export_reach_is_null_absent_and_true_when_carried():
    """A1's coarse ``output`` reach is never copied onto promotion/export substages.

    ``promoted`` and ``terminal`` have no dedicated A1 checkpoint, so an absent
    substage record serializes ``checkpoint_reached: null``
    (unattempted/unknown), never a fabricated ``false`` and never the coarse
    ``output`` reach; a carried substage record proves its own reach, and presence
    is explicit record existence, so a carried **empty** record is reach ``true``
    with an unavailable metric.  Rows whose evidence *is* the A1 output record
    (``raw_winner``, ``volume_cell``, candidates) keep the output reach.
    """
    reference = {(0, 0, 0): STONE}
    absent = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(), placement={(0, 0, 0): STONE},
        volume_cells={(0, 0, 0): STONE}, palette=(STONE,))
    rows = {row.checkpoint: row for row in
            absent.section[attribution.MetricSpace.EXACT_STATE].rows}
    # ``raw_winner``'s evidence is the A1 output-stage record, so it keeps output reach.
    assert rows["raw_winner"].checkpoint_reached is True
    assert rows["promoted"].reached is False
    assert rows["promoted"].checkpoint_reached is None
    assert rows["promoted"].to_record()["checkpoint_reached"] is None
    assert rows["terminal"].checkpoint_reached is None
    carried = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(), placement={(0, 0, 0): STONE},
        volume_cells={(0, 0, 0): STONE},
        promoted_cells={(0, 0, 0): STONE},
        decoded_terminal={(0, 0, 0): STONE}, palette=(STONE,))
    carried_rows = {row.checkpoint: row for row in
                    carried.section[attribution.MetricSpace.EXACT_STATE].rows}
    assert carried_rows["promoted"].reached is True
    assert carried_rows["promoted"].checkpoint_reached is True
    assert carried_rows["terminal"].reached is True
    assert carried_rows["terminal"].checkpoint_reached is True


def test_route_b_carried_empty_substage_record_proves_reach_with_a_null_metric():
    """Record presence is explicit ``is not None``, never mapping truthiness.

    A carried but empty promotion/decoded-export mapping proves the substage record
    exists, so its ``checkpoint_reached`` is ``true`` even though the empty record
    yields no metric (``reached: false``, ``metric_value: null``).  Only a missing
    record is ``null``.
    """
    reference = {(0, 0, 0): STONE}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(), placement={(0, 0, 0): STONE},
        volume_cells={(0, 0, 0): STONE},
        promoted_cells={}, decoded_terminal={}, palette=(STONE,))
    rows = {row.checkpoint: row for row in
            record.section[attribution.MetricSpace.EXACT_STATE].rows}
    for checkpoint in ("promoted", "terminal"):
        assert rows[checkpoint].reached is False
        assert rows[checkpoint].metric_value is None
        assert rows[checkpoint].checkpoint_reached is True
        assert rows[checkpoint].to_record()["checkpoint_reached"] is True


def test_checkpoint_reach_is_serialized_and_validated_independently_of_the_metric():
    """A row with no production checkpoint carries null reach; non-bool is refused."""
    row = attribution.label_metric_row(
        checkpoint="facade_grid", space=attribution.MetricSpace.GEOMETRY,
        kind="label", numerator=1, denominator=2, denominator_label="all visible V")
    assert row.reached is True
    assert row.checkpoint_reached is None
    assert row.to_record()["checkpoint_reached"] is None
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionRow(
            checkpoint="facade_grid", metric="geometry_recall",
            space=attribution.MetricSpace.GEOMETRY, kind="geometry",
            universe=attribution.REFERENCE_CELLS_UNIVERSE,
            denominator="V", reached=False, typed_reason="absent",
            checkpoint_reached="yes")
    with pytest.raises(attribution.AttributionRefusal):
        attribution.AttributionRow(
            checkpoint="facade_grid", metric="geometry_recall",
            space=attribution.MetricSpace.GEOMETRY, kind="geometry",
            universe=attribution.REFERENCE_CELLS_UNIVERSE,
            denominator="V", reached=False, typed_reason="absent",
            checkpoint_reached=1)


def test_evaluator_control_totals_are_never_attached_to_the_entry_row():
    """An alternate matcher's totals cannot become an achieved Route B P row.

    ``evaluate_scene`` / ``select_mutual_matches`` is a distinct matcher from the
    entry's ``joint_matching.match_and_solve_pair``.  Even with a complete
    aggregate supplied, an unbound source or a rederived digest that does not equal
    the accepted ``RelativeCameraEvidence.matches_sha256`` must leave the entry row
    typed unavailable with ``metric: null`` and no P number.
    """
    reference = {(0, 0, 0): STONE}
    control = _control_entry_evidence()
    assert control.bound is False
    assert control.reason == attribution.ENTRY_MATCH_UNBOUND_REASON
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,),
        entry_pair_evidence=control)
    section = record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE]
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.reached is False and row.metric_value is None
    assert row.typed_reason == attribution.ENTRY_MATCH_UNBOUND_REASON
    assert row.is_accuracy is False
    # A digest mismatch is equally unbound, even with the joint matcher's name.
    mismatched = attribution.EntryPairEvidence(
        binding=attribution.EntryMatchBinding(
            source=attribution.JOINT_MATCHER_SOURCE,
            matches_sha256="b" * 64, accepted_matches_sha256="c" * 64),
        totals=_pair_totals(truth_pairs=232, truth_pairs_returned=0,
                            returned=33, positive=0))
    assert mismatched.bound is False and mismatched.reason == \
        attribution.ENTRY_MATCH_UNBOUND_REASON
    with pytest.raises(attribution.AttributionRefusal):
        attribution.pair_metric_rows(
            checkpoint="entry", totals=mismatched.totals, binding=mismatched.binding)


def test_a_zero_truth_pair_denominator_is_null_not_a_measured_zero():
    """An empty truth denominator is a typed null through ``metric_value``."""
    reference = {(0, 0, 0): STONE}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(),
        placement={(0, 0, 0): STONE}, palette=(STONE,),
        entry_pair_evidence=_bound_entry_evidence(
            truth_pairs=0, truth_pairs_returned=0, returned=4, positive=3))
    rows = {row.metric: row
            for row in record.section[attribution.MetricSpace.PAIR_CORRESPONDENCE].rows}
    recall = rows["truth_pair_recall"]
    assert recall.reached is True
    assert recall.value is None
    assert recall.undefined_reason == "no_truth_keypoint_pairs"
    assert rows["positive_precision"].value == 3 / 4


def test_route_b_carries_both_geometry_checkpoints_with_an_unpartitioned_transition():
    """``placed`` and ``volume_cell`` are both G; the step reconciles on ``matched``.

    A real run can carry both geometry records.  Their transition must use
    unpartitioned same-universe geometry reconciliation (never the exact-state
    ``geometry_lost``/``state_lost`` partition), and ``next = previous + gained -
    lost`` must hold on the reported geometry success count.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): STONE, (2, 0, 0): STONE}
    placement = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): STONE}
    volume_cells = {(0, 0, 0): STONE, (1, 0, 0): DIRT}  # (2,0,0) lost from geometry
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(), visible_reference=_visible_reference(reference),
        reference=reference, alignment=_alignment(), placement=placement,
        volume_cells=volume_cells, palette=(STONE, DIRT))
    transitions = [item for item in record.section[attribution.MetricSpace.GEOMETRY].transitions
                   if item.previous_checkpoint == "placed"
                   and item.next_checkpoint == "volume_cell"]
    assert len(transitions) == 1
    step = transitions[0]
    assert step.space is attribution.MetricSpace.GEOMETRY
    assert step.universe == attribution.REFERENCE_CELLS_UNIVERSE
    assert step.previous == 3 and step.lost == 1 and step.gained == 0 and step.net == -1
    # Unpartitioned: the exact-state partition must stay null on a G transition.
    assert step.geometry_lost is None and step.state_lost is None
    assert step.to_record()["loss_partition"] is None
    # The reconciliation identity holds on the reported geometry success count.
    assert step.previous + step.gained - step.lost == 2


def test_route_b_exact_state_transitions_partition_loss_disjointly_while_g_stays_unpartitioned():
    """E-to-E Route B steps partition loss; only the G-to-G step is unpartitioned.

    ``raw_winner``, ``promoted`` and ``terminal`` are all exact-state checkpoints, so
    each E step must carry the disjoint ``geometry_lost``/``state_lost`` partition
    and reconcile ``next = previous + gained - lost``.  Only ``placed ->
    volume_cell`` is geometry, so only it stays unpartitioned.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): STONE, (2, 0, 0): STONE}
    placement = {(0, 0, 0): STONE, (1, 0, 0): STONE, (2, 0, 0): STONE}
    volume_cells = {(0, 0, 0): STONE, (1, 0, 0): STONE}   # (2,0,0) lost from geometry
    proposals = {
        (0, 0, 0): SimpleNamespace(state=STONE, candidate_costs=((STONE, 0.0),)),
        (1, 0, 0): SimpleNamespace(state=STONE, candidate_costs=((STONE, 0.0),)),
        (2, 0, 0): SimpleNamespace(state=STONE, candidate_costs=((STONE, 0.0),)),
    }
    # (2,0,0) leaves geometry and (1,0,0) keeps geometry but loses its state.
    promoted = {(0, 0, 0): STONE, (1, 0, 0): DIRT}
    # (0,0,0) then leaves geometry as well.
    terminal = {(1, 0, 0): DIRT}
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(), placement=placement, volume_cells=volume_cells,
        proposals=proposals, promoted_cells=promoted, decoded_terminal=terminal)
    exact = record.section[attribution.MetricSpace.EXACT_STATE]
    steps = {(item.previous_checkpoint, item.next_checkpoint): item
             for item in exact.transitions}
    assert set(steps) == {("raw_winner", "promoted"), ("promoted", "terminal")}
    for step in steps.values():
        assert step.space is attribution.MetricSpace.EXACT_STATE
        assert step.geometry_lost is not None and step.state_lost is not None
        assert step.geometry_lost + step.state_lost == step.lost
        assert step.net == step.gained - step.lost
    first = steps[("raw_winner", "promoted")]
    assert first.lost == 2 and first.geometry_lost == 1 and first.state_lost == 1
    assert first.to_record()["loss_partition"] is not None
    second = steps[("promoted", "terminal")]
    assert second.lost == 1 and second.geometry_lost == 1 and second.state_lost == 0
    geometry_steps = [item for item in
                      record.section[attribution.MetricSpace.GEOMETRY].transitions
                      if (item.previous_checkpoint, item.next_checkpoint) ==
                      ("placed", "volume_cell")]
    assert len(geometry_steps) == 1
    # Only the G-to-G step is unpartitioned.
    assert geometry_steps[0].geometry_lost is None
    assert geometry_steps[0].state_lost is None
    assert geometry_steps[0].to_record()["loss_partition"] is None


# ---------------------------------------------------------------------------
# Route B run-native candidate/alignment adapter
# ---------------------------------------------------------------------------

def _run_native_map(reference, turns, translation):
    """A run-frame candidate whose single permitted alignment is ``reference``."""
    inverse = _inverse_translation(translation, turns)
    inverse_turns = (-turns) % 4
    return {
        tuple(_unrot(coordinate, turns)[axis] + inverse[axis] for axis in range(3)):
            attribution._rotate_state(state, inverse_turns)
        for coordinate, state in reference.items()
    }


def test_align_candidate_rankings_transforms_cell_and_every_state_exactly_once():
    """A ranked proposal list aligns its cell and every state once, in order."""
    turns, translation = 1, (2, 3, -1)
    alignment = _alignment(turns, translation)
    run_cell = (0, 0, 0)
    aligned = attribution.align_candidate_rankings(
        {run_cell: (STAIRS_NORTH, STONE, DIRT)}, alignment)
    expected_cell = next(iter(attribution.align_candidate(
        {run_cell: STAIRS_NORTH}, alignment)))
    assert list(aligned) == [expected_cell]
    # Order (and any cost association) is preserved; the state rotates with the cell.
    assert aligned[expected_cell] == (STAIRS_EAST, STONE, DIRT)
    assert aligned[expected_cell][0] == attribution._rotate_state(STAIRS_NORTH, turns)
    # A second transform is a different cell/state, proving exactly-once semantics.
    double = attribution.align_candidate(
        {expected_cell: aligned[expected_cell][0]}, alignment)
    assert next(iter(double)) != expected_cell
    assert next(iter(double.values())).properties != dict(STAIRS_EAST.properties)
    assert attribution.align_candidate_rankings({}, alignment) == {}
    with pytest.raises(attribution.AttributionRefusal):
        attribution.align_candidate_rankings(
            {run_cell: (STAIRS_NORTH,)}, {**alignment, "scale": 2.0})


def test_coordinate_only_scorer_matches_the_accepted_visible_geometry_counts():
    """Coordinate-only G scoring uses the accepted ``visible_metrics`` counts.

    A geometry-only placement/volume record carries no state, so it is scored from
    coordinates alone.  The counts must equal what ``evaluation.visible_metrics``
    reports for the same aligned coordinates (matched, phantoms, hidden inferred),
    while ``exact`` stays empty because there is genuinely no label.
    """
    reference = {(0, 0, 0): STONE, (1, 0, 0): STAIRS_EAST, (2, 0, 0): DIRT}
    run_cells = {(0, 0, 0), (1, 0, 0), (1, 1, 0)}
    turns, translation = 1, (2, 0, -1)
    alignment = _alignment(turns, translation)
    score = attribution.score_aligned_geometry_coordinates(
        reference=reference, cells=run_cells,
        visible_reference=_visible_reference(reference), alignment=alignment,
        checkpoint="placed", space=attribution.MetricSpace.GEOMETRY)
    aligned = {coordinate: STONE
               for coordinate in attribution.align_candidate_coordinates(
                   run_cells, alignment)}
    expected = visible_metrics(reference, aligned, set(reference))
    assert len(score.matched) == expected.visible_coordinate_matches
    assert score.aligned_cells == expected.aligned_candidate_blocks
    assert score.phantom_cells == expected.phantom_blocks
    assert score.hidden_inferred_cells == expected.hidden_inferred_blocks
    assert score.exact == frozenset()
    assert score.geometry_recall == {
        "value": expected.visible_coordinate_matches / len(reference),
        "numerator": expected.visible_coordinate_matches,
        "denominator": len(reference), "undefined_reason": None}


def test_best_visible_alignment_for_coordinates_recovers_a_nonidentity_transform():
    """The coordinate-only search recovers the accepted nonidentity transform."""
    turns, translation = 1, (2, 0, -1)
    run_cells = {(0, 0, 1), (1, 0, 1), (0, 1, 1)}
    reference = {
        coordinate: STONE
        for coordinate in attribution.align_candidate_coordinates(
            run_cells, _alignment(turns, translation))}
    found = attribution.best_visible_alignment_for_coordinates(
        reference, run_cells, set(reference))
    assert found != (0, (0, 0, 0))
    assert attribution.align_candidate_coordinates(
        run_cells, _alignment(*found)) == frozenset(reference)


def test_geometry_cell_evidence_and_coordinate_alignment_refuse_invalid_input():
    """Coordinate-only evidence never accepts a state, scale or non-integer cell."""
    with pytest.raises(attribution.AttributionRefusal):
        attribution.GeometryCellEvidence(cells={(0, 0)})  # type: ignore[arg-type]
    with pytest.raises(attribution.AttributionRefusal):
        attribution.GeometryCellEvidence(cells=frozenset({(0, 0, "x")}))  # type: ignore[arg-type]
    with pytest.raises(attribution.AttributionRefusal):
        attribution.align_candidate_coordinates(
            {(0, 0, 0)}, {**_alignment(), "scale": 2.0})
    assert attribution.align_candidate_coordinates(
        {(0, 0, 0)}, _alignment(1, (2, 0, -1))) == frozenset({(2, 0, -1)})


def test_route_b_candidate_rows_emit_from_a_valid_c_context_with_directional_alignment():
    """Palette + proposals route records emit C rows, never a C-space refusal.

    ``placement``, the palette and every ranked proposal candidate are run-native
    and the alignment is a non-identity translation + Y-quarter-turn.  A single
    transform must map the coordinates and every directional candidate state, so
    the truth state is found at the rank it really holds: the C rows are measured
    over the same frozen visible universe as the G/E rows, and the ``candidate_k``
    sweep emits its metrics for multiple ``K`` instead of raising.  The palette is
    supplied run-native (``oak_stairs[facing=north]`` for a ``facing=east`` truth at
    quarter-turn 1), so an unrotated palette comparison would score 2/3, not 3/3.
    """
    reference = {(0, 0, 0): STAIRS_EAST, (1, 0, 0): STONE, (2, 0, 0): DIRT}
    turns, translation = 1, (2, 0, -1)
    run_map = _run_native_map(reference, turns, translation)
    inverse = _inverse_translation(translation, turns)
    run_cells = {
        reference_cell: tuple(
            _unrot(reference_cell, turns)[axis] + inverse[axis] for axis in range(3))
        for reference_cell in reference}
    # Truth is rank 2 at the first cell and rank 1 at the other two.
    desired = {
        (0, 0, 0): (STONE, STAIRS_EAST),
        (1, 0, 0): (STONE, DIRT),
        (2, 0, 0): (DIRT, GOLD),
    }
    inverse_turns = (-turns) % 4
    proposals = {}
    for reference_cell, candidates in desired.items():
        run_cell = run_cells[reference_cell]
        run_candidates = tuple(
            attribution._rotate_state(state, inverse_turns) for state in candidates)
        proposals[run_cell] = SimpleNamespace(
            cell=run_cell, state=run_candidates[0], status="proposed",
            candidate_costs=tuple((state, float(index))
                                  for index, state in enumerate(run_candidates)))
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(turns, translation), placement=run_map,
        # A run can only rank states its own palette holds, so the GOLD distractor
        # at the first cell's rank 2 must be part of the run-native palette.
        palette=(attribution._rotate_state(STAIRS_EAST, inverse_turns), STONE, DIRT,
                 GOLD),
        proposals=proposals, candidate_depths=(1, 2, 4))
    # The run-native placement scores full geometry only if aligned exactly once.
    placed = {row.metric: row for row in
              record.section[attribution.MetricSpace.GEOMETRY].rows
              if row.checkpoint == "placed"}
    assert placed["geometry_recall"].numerator == 3
    assert placed["geometry_recall"].denominator_count == 3
    candidate = record.section[attribution.MetricSpace.CANDIDATE]
    assert candidate.rows and all(row.reached for row in candidate.rows)
    by_metric = {row.metric: row for row in candidate.rows}
    assert by_metric["palette_coverage"].numerator == 3
    assert by_metric["palette_coverage"].value == 1.0
    # Exactly-once coordinate and directional-state alignment: K=1 finds the two
    # rank-1 truths, and K=2 additionally finds the rank-2 truth.  An unrotated
    # state list would cap K=2 at 2; unaligned coordinates would leave it at 0.
    assert by_metric["candidate_available_k1"].numerator == 2
    assert by_metric["candidate_available_k1"].denominator_count == 3
    assert by_metric["candidate_available_k2"].numerator == 3
    assert by_metric["candidate_available_k4"].numerator == 3


def test_run_native_palette_rotation_keeps_candidate_and_exact_state_consistent():
    """A run-native directional palette aligns once; an outside state stays charged.

    Before this repair the run-native ``oak_stairs`` palette was compared unrotated
    against the reference-frame truth under the non-identity quarter-turn below, so
    the C palette row read 2/4 while the candidate and E rows read 3/4.  The palette
    now receives the same quarter-turn as the candidate, so the three aligned
    in-palette truth cells are covered consistently and the deliberate
    out-of-palette ``GOLD`` cell keeps the full ``|V|`` denominator.
    """
    reference = {
        (0, 0, 0): STAIRS_EAST,
        (1, 0, 0): STONE,
        (2, 0, 0): DIRT,
        (3, 0, 0): GOLD,
    }
    turns, translation = 3, (-2, 4, 1)
    inverse_turns = (-turns) % 4
    inverse = _inverse_translation(translation, turns)
    run_cells = {
        cell: tuple(_unrot(cell, turns)[axis] + inverse[axis] for axis in range(3))
        for cell in reference}
    # A run can only place/propose states its palette holds: the GOLD truth cell is
    # deliberately given a DIRT candidate, so the palette charge is observable.
    aligned_run_map = dict(reference)
    aligned_run_map[(3, 0, 0)] = DIRT
    placement = {
        run_cells[cell]: attribution._rotate_state(state, inverse_turns)
        for cell, state in aligned_run_map.items()}
    proposals = {
        run_cells[cell]: SimpleNamespace(
            cell=run_cells[cell],
            state=attribution._rotate_state(state, inverse_turns),
            status="proposed",
            candidate_costs=(
                (attribution._rotate_state(state, inverse_turns), 0.0),))
        for cell, state in aligned_run_map.items()}
    # Run-native palette: the inverse rotation of the reference directional state;
    # GOLD is deliberately absent so the out-of-palette denominator is exercised.
    run_palette = tuple(
        attribution._rotate_state(state, inverse_turns)
        for state in (STAIRS_EAST, STONE, DIRT))
    record = attribution.route_b_attribution(
        frontier=_synthetic_route_b_frontier(full_tail=True),
        visible_reference=_visible_reference(reference), reference=reference,
        alignment=_alignment(turns, translation), placement=placement,
        palette=run_palette, proposals=proposals, candidate_depths=(1,))
    candidate = {row.metric: row for row in
                 record.section[attribution.MetricSpace.CANDIDATE].rows}
    # Palette and candidate availability agree: 3 in-palette truth cells covered,
    # the out-of-palette GOLD cell charged to the palette, denominator still |V|.
    assert candidate["palette_coverage"].numerator == 3
    assert candidate["palette_coverage"].denominator_count == 4
    assert candidate["palette_coverage_in_palette"].numerator == 3
    assert candidate["palette_coverage_in_palette"].denominator_count == 3
    assert candidate["candidate_available_k1"].numerator == 3
    assert candidate["candidate_available_k1"].denominator_count == 4
    # E consistency: the same three in-palette cells recover their exact state.
    exact = {row.metric: row for row in
             record.section[attribution.MetricSpace.EXACT_STATE].rows
             if row.checkpoint == "raw_winner"}
    assert exact["exact_state_recovery"].numerator == 3
    assert exact["exact_state_recovery"].denominator_count == 4
    assert exact["exact_state_recovery_in_palette"].numerator == 3
    assert exact["exact_state_recovery_in_palette"].denominator_count == 3


# ---------------------------------------------------------------------------
# Route B runner: real production record adapters
# ---------------------------------------------------------------------------

def _real_placement(coordinates, *, frame="f0"):
    """A real ``ResolvedFacePlacement`` emitting exactly ``coordinates``.

    Each coordinate is placed on the ``+Z`` normal (axis 2, sign +1) with tangent
    axes 0 (column -> x) and 1 (row -> y), so the record's own face evidence derives
    the same world coordinate production's ``build_plane_volume`` derives
    (``reconstruction/plane_volume.py``).  Coordinates must have non-negative x/y;
    distinct z values use one face each.  No state is fabricated: placement is
    genuinely geometry-only.
    """
    by_normal = {}
    for coordinate in coordinates:
        coordinate = tuple(coordinate)
        if len(coordinate) != 3 or coordinate[0] < 0 or coordinate[1] < 0:
            raise ValueError("the placement helper needs non-negative x/y triples")
        by_normal.setdefault(coordinate[2], []).append(coordinate)
    faces = []
    for index, normal_block in enumerate(sorted(by_normal)):
        cells = tuple(sorted(
            (placement_mod._make_cell(row=cell[1], column=cell[0],
                                      lateral_u=cell[0], lateral_v=cell[1])
             for cell in by_normal[normal_block]),
            key=lambda cell: (cell.row, cell.column, cell.cell_sha256)))
        faces.append(placement_mod.ResolvedFace.build(
            frame=frame, plane_index=index, reading_axis=2, absolute_quadrant=0,
            source_region_id="a" * 64, reading_identity_sha256="b" * 64,
            support_sha256="c" * 64,
            world_normal_axis=2, world_normal_sign=1, c=(1, 1), d=(1, 1),
            translation=((0, 1), (0, 1), (normal_block - 1, 1)),
            alpha=((0, 1), (0, 1), (0, 1)),
            face_gauge=(normal_block, 1), face=normal_block,
            outward_sign=-1, normal_block=normal_block,
            tangent_mappings=(
                placement_mod._make_tangent_mapping(
                    local_axis="u", local_label=0, world_axis=0, world_sign=1,
                    origin=(0, 1), anchor=0),
                placement_mod._make_tangent_mapping(
                    local_axis="v", local_label=1, world_axis=1, world_sign=1,
                    origin=(0, 1), anchor=0)),
            cells=cells))
    work = placement_mod._make_work({
        "faces": len(faces), "supports_checked": len(faces),
        "support_cells": sum(len(face.cells) for face in faces),
        "anchors": sum(len(face.tangent_mappings) + 2 * len(face.cells)
                       for face in faces)})
    return placement_mod.ResolvedFacePlacement.build(
        schema_version=placement_mod.V2_PLACEMENT_SCHEMA,
        producer_version=placement_mod.V2_PLACEMENT_PRODUCER_VERSION,
        gauge_state=placement_mod.V2_PLACEMENT_GAUGE_STATE,
        gauge_axes=placement_mod.V2_PLACEMENT_WORLD_AXES,
        alpha=((0, 1), (0, 1), (0, 1)),
        component_bindings=({"component": "c0"},),
        faces=tuple(faces), work=work)


def _real_volume(coordinates):
    """A real ``PlaneVolumeResult`` whose cells carry exactly ``coordinates``."""
    coordinates = tuple(sorted(coordinates))
    claims = [
        volume_mod.CellClaim.build(
            frame="f0", plane_index=0, reading_axis=0, absolute_quadrant=0,
            source_region_id="a" * 64, reading_identity_sha256="b" * 64,
            support_sha256="c" * 64, face_sha256="d" * 64,
            row=index, column=0, world_normal_axis=0, world_normal_sign=-1,
            outward_sign=-1, face=coordinate[0], normal_block=coordinate[0],
            tangent_u_axis=1, lateral_u=coordinate[1],
            tangent_v_axis=2, lateral_v=coordinate[2], coordinate=coordinate)
        for index, coordinate in enumerate(coordinates)]
    cells = tuple(
        volume_mod.SupportedCellObservation.build(
            coordinate=coordinate, claims=(claim,), frame_ids=("f0",),
            conflict_kind=None)
        for coordinate, claim in zip(coordinates, claims))
    bindings = ({
        "frame": "f0", "plane_index": 0, "reading_axis": 0,
        "reading_identity_sha256": "b" * 64, "support_sha256": "c" * 64,
        "face_sha256": "d" * 64, "source_region_id": "a" * 64,
        "alternative_distance": [0, 1], "phase_anchor_sha256": "e" * 64,
        "support_cells": len(cells), "supported_cells": len(cells),
        "unknown_cells": 0,
    },)
    low = tuple(min(coordinate[axis] for coordinate in coordinates)
                for axis in range(3))
    high = tuple(max(coordinate[axis] for coordinate in coordinates)
                 for axis in range(3))
    dense = 1
    for axis in range(3):
        dense *= high[axis] - low[axis] + 1
    work = volume_mod._make_work({
        "faces": 1, "support_cells": len(cells),
        "contributions": len(cells), "coordinates": len(cells),
        "conflicts": 0, "omitted_unknown": 0, "dense_bounds_cells": dense})
    return volume_mod.PlaneVolumeResult.build(
        schema_version=volume_mod.V3_VOLUME_SCHEMA,
        producer_version=volume_mod.V3_VOLUME_PRODUCER_VERSION,
        gauge_state=volume_mod.V3_VOLUME_GAUGE_STATE, gauge_axes=(0, 1, 2),
        alpha=((0, 1), (0, 1), (0, 1)), placement_sha256="f" * 64,
        support_bindings=bindings, cells=cells, conflicts=(), omitted=(),
        cell_bounds=(low, high), dense_bounds_cells=dense,
        normalization_offset=low, export_bounds=None, complete=False, work=work)


def _real_promotion(states):
    """A real ``PromotionResult`` built through the ``PromotedBlock`` constructor."""
    blocks = tuple(
        promotion_mod.PromotedBlock.build(
            coordinate=coordinate, state_name=state.name,
            state_properties=tuple(state.properties), confidence=1.0,
            frame_ids=("f0",), provenance=(), observation_digests=("a" * 64,),
            proposal_digest="b" * 64, evidence_digest="c" * 64)
        for coordinate, state in sorted(states.items()))
    return promotion_mod.PromotionResult.build(
        schema_version=promotion_mod.PROMOTION_SCHEMA,
        producer_version=promotion_mod.PROMOTION_PRODUCER_VERSION,
        volume_sha256="d" * 64, proposal_result_digest="e" * 64,
        gauge_state=promotion_mod.PROMOTION_GAUGE_STATE,
        gauge_axes=promotion_mod.PROMOTION_GAUGE_AXES,
        scale_state=promotion_mod.PROMOTION_SCALE_STATE,
        normalization_state=promotion_mod.PROMOTION_NORMALIZATION_STATE,
        normalization_offset=None, export_bounds=None, omitted_unknown=0,
        cells=blocks, export_ready=True, exact_state_eligible=True,
        max_cells=100, max_proposals=100, max_payload_bytes=1_000_000)


class _RunnerOutcome:
    """A minimal synthetic ``ImageDerivedProductionResult`` seam."""

    def __init__(self, *, placement=None, volume=None, promotion=None,
                 palette=(STONE, DIRT), proposals=None):
        self.stage = "output"
        self.status = "promoted"
        self.reason = ""
        self.detail = ""
        self.entry_sha256 = "a" * 64
        self.lattice_sha256 = "c" * 64
        self.geometry_sha256 = "d" * 64
        self.output_sha256 = "e" * 64
        self.result_sha256 = "f" * 64
        self.entry = SimpleNamespace(graph=None)
        self.lattice = None
        self.geometry = SimpleNamespace(status="assembled", universe=object())
        self.output = SimpleNamespace(
            policy=SimpleNamespace(palette=list(palette)),
            placement=placement, volume=volume, promotion=promotion,
            proposals=proposals)


def _drive_route_b_runner(module_name, monkeypatch, *, outcome, reference,
                          alignment=(0, (0, 0, 0)), capture=None,
                          measure_alignment=False):
    """Run the real Route B runner seam over synthetic production records.

    Production and scoring are executed separately exactly as ``main`` does, with
    the production entry fabricated and no fixture, image or truth path opened.
    """
    module = _load_runner(module_name, RUNNER)
    envelope = SimpleNamespace(
        stage=outcome.stage, status=outcome.status, reason=outcome.reason,
        detail=outcome.detail, record_sha256="9" * 64, session_sha256="b" * 64,
        entry_sha256=outcome.entry_sha256, lattice_sha256=outcome.lattice_sha256,
        geometry_sha256=outcome.geometry_sha256,
        output_sha256=outcome.output_sha256)
    monkeypatch.setattr(module, "derive_image_output_from_paths",
                        lambda sources, project_path: outcome)
    monkeypatch.setattr(module, "ImageDerivedProductionResult", type(outcome))
    monkeypatch.setattr(
        module, "route_b_frontier",
        lambda **kwargs: _synthetic_route_b_frontier(
            geometry_stop=outcome.output is None,
            full_tail=outcome.output is not None))
    monkeypatch.setattr(module, "_persisted_envelope", lambda target: envelope)
    monkeypatch.setattr(
        module, "route_b_evaluator_pair_control",
        lambda truth: {"totals": None, "reason": "synthetic_control",
                       "coverage": {}})
    monkeypatch.setattr(module, "_truth_summary", lambda truth: {})

    def measured(reference_arg, coordinates, visible):
        if capture is not None:
            capture["coordinates"] = frozenset(coordinates)
        return alignment

    if not measure_alignment:
        monkeypatch.setattr(module, "best_visible_alignment_for_coordinates", measured)
    captured = {}
    real_builder = attribution.route_b_attribution

    def spy(**kwargs):
        captured.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(module, "route_b_attribution", spy)
    truth = {"reference": reference, "visible_reference": _visible_reference(reference)}
    produced = module._route_b_once([Path("frame.png")], Path("unused.mcscan"))
    summary = module._score_route_b(outcome=produced, truth=truth)
    return captured, summary


def _serialized_checkpoint_row(summary, checkpoint, metric=None):
    rows = [row for section in summary["attribution"]["sections"]
            for row in section["rows"] if row["checkpoint"] == checkpoint
            and (metric is None or row["metric"] == metric)]
    assert rows, (checkpoint, metric)
    return rows[0]


def test_runner_reads_real_placement_volume_and_promotion_records(monkeypatch):
    """The runner adapts the real typed records, never a ``{coordinate: None}`` map."""
    reference = {(0, 0, 1): STONE, (1, 0, 1): DIRT, (0, 1, 1): STONE}
    outcome = _RunnerOutcome(
        placement=_real_placement(reference), volume=_real_volume(reference),
        promotion=_real_promotion(reference), palette=(STONE, DIRT))
    captured, summary = _drive_route_b_runner(
        "_a2_runner_real_records", monkeypatch, outcome=outcome, reference=reference)
    # Real coordinate-only geometry evidence reaches the builder, never a state map.
    assert isinstance(captured["placement"], attribution.GeometryCellEvidence)
    assert set(captured["placement"].cells) == set(reference)
    assert isinstance(captured["volume_cells"], attribution.GeometryCellEvidence)
    assert set(captured["volume_cells"].cells) == set(reference)
    assert captured["promoted_cells"] == dict(reference)
    assert summary["production_records"] == {
        "placement_cells": 3, "volume_cells": 3, "promoted_cells": 3,
        "proposal_cells": 0}
    placed = _serialized_checkpoint_row(summary, "placed", "geometry_recall")
    volume_row = _serialized_checkpoint_row(summary, "volume_cell", "geometry_recall")
    assert placed["metric_value"] == {"value": 1.0, "numerator": 3,
                                      "denominator": 3, "undefined_reason": None}
    assert volume_row["metric_value"]["numerator"] == 3
    promoted = _serialized_checkpoint_row(summary, "promoted", "exact_state_recovery")
    assert promoted["metric_value"]["numerator"] == 3
    assert promoted["checkpoint_reached"] is True


def test_runner_measures_alignment_from_run_native_geometry_and_aligns_once(monkeypatch):
    """A nonidentity measured alignment is applied exactly once to real coordinates."""
    turns, translation = 1, (2, 0, -1)
    run_cells = {(0, 0, 1), (1, 0, 1), (0, 1, 1)}
    alignment = _alignment(turns, translation)
    reference = {
        next(iter(attribution.align_candidate_coordinates({cell}, alignment))): STONE
        for cell in run_cells}
    outcome = _RunnerOutcome(
        placement=_real_placement(run_cells), volume=None, promotion=None,
        palette=(STONE,))
    capture = {}
    captured, summary = _drive_route_b_runner(
        "_a2_runner_nonidentity", monkeypatch, outcome=outcome, reference=reference,
        alignment=(turns, translation), measure_alignment=True)
    # The production seam carries run-native coordinates; the real search recovers
    # their nonidentity transform and the builder applies it once.
    assert isinstance(captured["placement"], attribution.GeometryCellEvidence)
    assert set(captured["placement"].cells) == run_cells
    placed = _serialized_checkpoint_row(summary, "placed", "geometry_recall")
    assert placed["metric_value"]["numerator"] == 3
    assert placed["metric_value"]["denominator"] == 3
    assert summary["alignment"]["rotation_quarter_turns"] == turns
    assert summary["alignment"]["translation"] == list(translation)


@pytest.mark.parametrize("source", ["promoted_states", "proposals"])
def test_runner_alignment_falls_back_to_state_bearing_coordinate_keys(source):
    """Later state-bearing records also carry usable coordinates for alignment."""
    module = _load_runner(f"_a2_runner_alignment_{source}", RUNNER)
    turns, translation = 3, (-2, 1, 4)
    run_cells = {(0, 0, 1), (2, 0, 1), (0, 1, 1)}
    target = _alignment(turns, translation)
    reference = {
        next(iter(attribution.align_candidate_coordinates({cell}, target))): STONE
        for cell in run_cells}
    promoted = {cell: STONE for cell in run_cells} if source == "promoted_states" else None
    proposals = {cell: object() for cell in run_cells} if source == "proposals" else None

    measured = module._measured_route_b_alignment(
        reference=reference, visible_reference=_visible_reference(reference),
        placement_evidence=None, volume_evidence=None,
        promoted_states=promoted, proposals=proposals)
    assert measured["rotation_quarter_turns"] == turns
    assert measured["translation"] == list(translation)


def test_runner_without_coordinate_tail_preserves_typed_stage_ledger(monkeypatch):
    """No candidate means no alignment; frontier stages remain typed unavailable."""
    reference = {(0, 0, 1): STONE, (1, 0, 1): DIRT}
    outcome = _RunnerOutcome()
    outcome.output = None
    outcome.stage = "geometry"
    outcome.status = "unavailable"
    outcome.reason = "no_candidate_cells"
    outcome.output_sha256 = ""
    module = _load_runner("_a2_runner_no_geometry_tail", RUNNER)
    monkeypatch.setattr(module, "derive_image_output_from_paths",
                        lambda sources, project_path: outcome)
    monkeypatch.setattr(module, "ImageDerivedProductionResult", type(outcome))
    monkeypatch.setattr(
        module, "route_b_frontier",
        lambda **kwargs: _synthetic_route_b_frontier(geometry_stop=True))
    envelope = SimpleNamespace(
        stage=outcome.stage, status=outcome.status, reason=outcome.reason,
        detail=outcome.detail, record_sha256="9" * 64, session_sha256="b" * 64,
        entry_sha256=outcome.entry_sha256, lattice_sha256=outcome.lattice_sha256,
        geometry_sha256=outcome.geometry_sha256,
        output_sha256=outcome.output_sha256)
    monkeypatch.setattr(module, "_persisted_envelope", lambda target: envelope)
    monkeypatch.setattr(
        module, "best_visible_alignment_for_coordinates",
        lambda *args, **kwargs: pytest.fail("alignment must not run without candidates"))
    monkeypatch.setattr(module, "route_b_evaluator_pair_control",
                        lambda truth: {"totals": None, "reason": "unavailable",
                                       "coverage": {}})
    monkeypatch.setattr(module, "_truth_summary", lambda truth: {})
    produced = module._route_b_once([Path("frame.png")], Path("unused.mcscan"))
    truth = {"reference": reference,
             "visible_reference": _visible_reference(reference)}
    summary = module._score_route_b(outcome=produced, truth=truth)

    assert summary["attribution_available"] is True
    assert summary["alignment"] is None
    assert summary["envelope_sha256"] == "9" * 64
    assert summary["session_sha256"] == "b" * 64
    assert summary["result_sha256"] == outcome.result_sha256
    assert summary["terminal"] == {
        "stage": "geometry", "status": "unavailable",
        "reason": "no_candidate_cells", "detail": ""}
    assert summary["attribution"]["frontier"]["route"] == "route_b"
    assert summary["production_records"] == {
        "placement_cells": 0, "volume_cells": 0, "promoted_cells": 0,
        "proposal_cells": 0}
    for stage in ("placed", "volume_cell", "raw_winner", "promoted", "terminal"):
        row = _serialized_checkpoint_row(summary, stage)
        assert row["metric_value"] is None
        assert row["typed_reason"]
    for stage in ("palette", "candidate_k"):
        row = _serialized_checkpoint_row(summary, stage)
        assert row["metric_value"] is None
        assert row["typed_reason"]
    entry = _serialized_checkpoint_row(summary, "entry", "truth_pair_recall")
    assert entry["metric_value"] is None


def test_runner_carried_empty_promotion_keeps_reach_true_through_the_adapter(monkeypatch):
    """A carried promotion with zero cells proves substage reach at the runner seam.

    ``_promoted_states`` returns ``{}`` both for an absent promotion and for a
    carried promotion with no cells, and the runner's old ``promoted_states or
    None`` collapsed both to ``None``, erasing the reach evidence before the
    builder could distinguish them.  The builder must receive an **empty mapping**
    (never ``None``) so the serialized ``promoted`` row reports
    ``checkpoint_reached: true`` beside ``reached: false`` and a null metric: reach
    and metric availability are separate facts.
    """
    reference = {(0, 0, 1): STONE}
    outcome = _RunnerOutcome(
        placement=_real_placement(reference), volume=None,
        promotion=_real_promotion({}), palette=(STONE,))
    captured, summary = _drive_route_b_runner(
        "_a2_runner_promotion_carried_empty", monkeypatch, outcome=outcome,
        reference=reference)
    assert captured["promoted_cells"] is not None
    assert captured["promoted_cells"] == {}
    promoted = _serialized_checkpoint_row(summary, "promoted")
    assert promoted["checkpoint_reached"] is True
    assert promoted["reached"] is False
    assert promoted["metric_value"] is None
    assert promoted["typed_reason"]
    # The runner carries no decoded terminal record, so that substage stays unknown.
    terminal = _serialized_checkpoint_row(summary, "terminal")
    assert terminal["checkpoint_reached"] is None
    assert terminal["reached"] is False
    assert terminal["metric_value"] is None


def test_runner_absent_promotion_serializes_null_reach_not_false(monkeypatch):
    """An absent promotion record is ``null`` reach, never a reach-erasing ``{}``.

    The runner must pass ``None`` only when ``outcome.output.promotion is None``;
    the builder then serializes ``checkpoint_reached: null`` (unattempted/unknown,
    never a fabricated ``false``) with the metric still unavailable.
    """
    reference = {(0, 0, 1): STONE}
    outcome = _RunnerOutcome(
        placement=_real_placement(reference), volume=None, promotion=None,
        palette=(STONE,))
    captured, summary = _drive_route_b_runner(
        "_a2_runner_promotion_absent", monkeypatch, outcome=outcome,
        reference=reference)
    assert captured["promoted_cells"] is None
    promoted = _serialized_checkpoint_row(summary, "promoted")
    assert promoted["checkpoint_reached"] is None
    assert promoted["reached"] is False
    assert promoted["metric_value"] is None
    terminal = _serialized_checkpoint_row(summary, "terminal")
    assert terminal["checkpoint_reached"] is None
    assert terminal["reached"] is False
    assert terminal["metric_value"] is None


def test_runner_runs_production_before_evaluator_pins_and_truth(monkeypatch, tmp_path):
    """Both routes run production before the evaluator pin manifest or truth loads.

    ``main`` used to open the evaluator pin manifest and the frozen
    reference/scene/cameras/masks before it called the production route, so
    evaluator files could precede (and configure) reconstruction.  The required
    order is production call < pin loader < truth loader; the runner stubs record
    each call so a regression that reorders any loader fails here.
    """
    module = _load_runner("_a2_runner_order", RUNNER)
    calls = []

    def production(*args, **kwargs):
        calls.append("production")
        return (object(), None)

    def pins(path, role):
        calls.append("pins")
        return {"reference_sha256": "", "frames": [], "usable_masks": []}

    def truth(**kwargs):
        calls.append("truth")
        raise attribution.AttributionRefusal("truth-sentinel")

    monkeypatch.setattr(module, "run_route_a", production)
    monkeypatch.setattr(module, "run_route_b", production)
    monkeypatch.setattr(module, "load_pins", pins)
    monkeypatch.setattr(module, "load_frozen_truth", truth)
    first = tmp_path / "frame.png"
    first.write_bytes(b"frozen-stand-in")
    second = tmp_path / "frame-2.png"
    second.write_bytes(b"frozen-stand-in-2")
    reference = str(tmp_path / "ref.litematic")
    scene = str(tmp_path / "scene.json")
    cameras = str(tmp_path / "cameras.json")
    pin_manifest = str(tmp_path / "pins.json")

    exit_code = module.main([
        "--route", "route-a", "--image", str(first), "--reference", reference,
        "--scene", scene, "--cameras", cameras,
        "--pins", pin_manifest, "--fixture-role", "role"])
    assert exit_code == 2
    assert calls == ["production", "pins", "truth"]

    calls.clear()
    exit_code = module.main([
        "--route", "route-b", "--sources", str(first), str(second),
        "--reference", reference, "--scene", scene, "--cameras", cameras,
        "--pins", pin_manifest, "--fixture-role", "role"])
    assert exit_code == 2
    assert calls == ["production", "pins", "truth"]


# ---------------------------------------------------------------------------
# Determinism and digest binding
# ---------------------------------------------------------------------------

def test_identical_inputs_serialize_byte_identically_with_stable_digests():
    first = _route_a_attribution()
    second = _route_a_attribution()
    assert first.to_record() == second.to_record()
    assert attribution.attribution_digest(first) == attribution.attribution_digest(second)
    assert first.frontier.to_record() == second.frontier.to_record()


def test_canonical_json_is_key_sorted_and_rejects_non_finite_values():
    payload = {"b": 1, "a": [1, 2, 3]}
    assert attribution._canonical(payload) == '{"a":[1,2,3],"b":1}'
    with pytest.raises(attribution.AttributionRefusal):
        attribution._canonical({"x": float("nan")})
    with pytest.raises(attribution.AttributionRefusal):
        attribution._canonical({"x": float("inf")})


# ---------------------------------------------------------------------------
# Synthetic route-B tail from the real public constructors
# ---------------------------------------------------------------------------

def _recipe_tail():
    coordinate = (0, 0, 0)
    claim = volume_mod.CellClaim.build(
        frame="f0", plane_index=0, reading_axis=0, absolute_quadrant=0,
        source_region_id="a" * 64, reading_identity_sha256="b" * 64,
        support_sha256="c" * 64, face_sha256="d" * 64,
        row=0, column=0, world_normal_axis=0, world_normal_sign=-1,
        outward_sign=-1, face=0, normal_block=0, tangent_u_axis=1, lateral_u=0,
        tangent_v_axis=2, lateral_v=0, coordinate=coordinate)
    cell = volume_mod.SupportedCellObservation.build(
        coordinate=coordinate, claims=(claim,), frame_ids=("f0",), conflict_kind=None)
    bindings = tuple(sorted(({
        "frame": claim.frame, "plane_index": 0, "reading_axis": 0,
        "reading_identity_sha256": claim.reading_identity_sha256,
        "support_sha256": claim.support_sha256, "face_sha256": claim.face_sha256,
        "source_region_id": claim.source_region_id,
        "alternative_distance": [0, 1], "phase_anchor_sha256": "e" * 64,
        "support_cells": 1, "supported_cells": 1, "unknown_cells": 0,
    },), key=volume_mod._binding_sort_key))
    work = volume_mod._make_work({
        "faces": 1, "support_cells": 1, "contributions": 1, "coordinates": 1,
        "conflicts": 0, "omitted_unknown": 0, "dense_bounds_cells": 1})
    volume = volume_mod.PlaneVolumeResult.build(
        schema_version=volume_mod.V3_VOLUME_SCHEMA,
        producer_version=volume_mod.V3_VOLUME_PRODUCER_VERSION,
        gauge_state=volume_mod.V3_VOLUME_GAUGE_STATE, gauge_axes=(0, 1, 2),
        alpha=((0, 1), (0, 1), (0, 1)), placement_sha256="f" * 64,
        support_bindings=bindings, cells=(cell,), conflicts=(), omitted=(),
        cell_bounds=((0, 0, 0), (0, 0, 0)), dense_bounds_cells=1,
        normalization_offset=(0, 0, 0), export_bounds=None, complete=False,
        work=work)
    observation = FaceObservation(
        cell=coordinate, slot="north", frame_id="f0", normal=NORMAL_OF_SLOT["north"],
        patch=np.full((4, 4, 3), 125.0, dtype=np.float32),
        usable_coverage=1.0, support_frames=("f0",), provenance=("cam/a",))
    proposals = proposals_mod.propose_face_states(
        (observation,),
        (PaletteColor(BlockState("minecraft:stone"), (125, 125, 125)),),
        max_cost=1e9, min_margin=0.0, max_observations=1000,
        max_candidate_evaluations=100_000, max_payload_bytes=10_000_000)
    return volume, proposals


def test_synthetic_tail_from_real_constructors_drives_candidate_and_winner_rows():
    volume, proposals = _recipe_tail()
    assert proposals.status == proposals_mod.PROPOSED
    proposal = proposals.proposals[0] if hasattr(proposals, "proposals") \
        else proposals.cells[0]
    reference = {proposal.cell: proposal.state}
    score = _score(reference, reference,
                   space=attribution.MetricSpace.CANDIDATE,
                   palette=tuple(state for state, _cost in proposal.candidate_costs),
                   checkpoint="candidate_k")
    rows = attribution.candidate_metric_rows(
        score, depths=(1, 2), reachable_by_coordinate={
            proposal.cell: tuple(state for state, _cost in proposal.candidate_costs)})
    assert rows[0].numerator == 1  # the truth state is the argmin candidate
    winner_score = _score(reference, reference,
                          space=attribution.MetricSpace.EXACT_STATE,
                          palette=tuple(state for state, _cost
                                        in proposal.candidate_costs),
                          checkpoint="raw_winner")
    winner_rows = attribution.exact_state_metric_rows(winner_score, kind="label")
    assert winner_rows[0].value == 1.0
    assert volume.volume_sha256
    assert proposal.proposal_digest


# ---------------------------------------------------------------------------
# Tracked mask specifications and mandatory reviewed pins
# ---------------------------------------------------------------------------

MASK_ROOT = REPO_ROOT / "docs" / "evaluation_masks"
TRACKED_TEMPLATE_2 = MASK_ROOT / "templates-2"
PINNED_TEMPLATE_2 = {
    "2026-09-02_13.22.14.png":
        "c1536420ce3f3b7fe9c4cf92a2221313b712542752c20dcec51184dd0ca08245",
    "2026-09-02_13.24.24.png":
        "dfe16f05ded6a89bd5a6d354798afb0757baf26787d2b174dea184f7b593f685",
    "2026-09-02_13.24.33.png":
        "f8260d647a2bb338eda1b8deee09bdad06aaada5ac07e6850bf9fe876fe89b31",
}
_TRACKED_MASK_SIZE = (2560, 1369)


def _pair_runner():
    return _load_runner("_a2_report_runner_pair_control", RUNNER)


def test_evaluator_pair_control_is_typed_unavailable_without_frozen_coverage():
    """The runner's control seam fails typed, never fabricating a denominator.

    These synthetic truth mappings exercise only the fail-closed branches; they
    read no fixture, image, camera manifest or ground truth.  The control is never
    the Route B achieved P row.
    """
    module = _pair_runner()
    with pytest.raises(attribution.AttributionRefusal):
        module.route_b_evaluator_pair_control({})
    one_frame = {"frame_paths": {"a.png": Path("a.png")},
                 "scene_transform": SimpleNamespace(to_world=lambda cell: cell)}
    assert module.route_b_evaluator_pair_control(one_frame)["reason"] == \
        "entry_pair_frames_below_two"
    two_frames = {"frame_paths": {"a.png": Path("a.png"), "b.png": Path("b.png")},
                  "scene_transform": SimpleNamespace(to_world=lambda cell: cell)}
    assert module.route_b_evaluator_pair_control(two_frames)["reason"] == \
        "entry_pair_camera_missing"
    with_cameras = dict(two_frames, cameras={"a.png": object(), "b.png": object()})
    assert module.route_b_evaluator_pair_control(with_cameras)["reason"] == \
        "entry_pair_reference_empty"


def test_entry_match_context_reports_counts_and_digests_without_identities():
    """Retained match evidence is context only; no pixel-pair identities exist."""
    module = _pair_runner()
    evidence = SimpleNamespace(
        left_id="a", right_id="b", status="solved", match_count=33,
        mutual_survivors=12, matches_sha256="d" * 64, evidence_sha256="e" * 64)
    context = module.entry_match_context(
        SimpleNamespace(graph=SimpleNamespace(status="ok", evidence=(evidence,))))
    assert context["graph_status"] == "ok"
    assert context["identities_retained"] is False
    assert context["pair_count"] == 1
    assert context["match_count_total"] == 33
    assert context["matches_sha256"] == ["d" * 64]
    assert context["pairs"][0]["match_count"] == 33
    assert module.entry_match_context(SimpleNamespace(graph=None))["pair_count"] == 0


def _mask_runner():
    return _load_runner("_a2_report_runner_mask_provenance", RUNNER)

def _write_spec(path, *, rectangles=(), polygons=(), size=_TRACKED_MASK_SIZE):
    path.write_text(json.dumps({
        "schema": 2, "image_size": [size[0], size[1]],
        "excluded_rectangles": [list(item) for item in rectangles],
        "excluded_polygons": [[list(point) for point in polygon]
                              for polygon in polygons],
    }), encoding="utf-8")


def _fake_frame(size=_TRACKED_MASK_SIZE):
    return np.zeros((size[1], size[0]), dtype=np.uint8)


def _write_frame_png(path, size):
    """Write a synthetic all-zero frame PNG of the given ``(width, height)``."""
    from PIL import Image

    Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8), mode="L").save(path)


def test_tracked_mask_specs_reproduce_every_reviewed_pin_domain():
    """The tracked JSON specifications are the pinned source of truth.

    This reads only the three tracked specification JSON files and a synthetic
    all-zero frame of the declared size; it opens no screenshot, reference, mask
    PNG, camera manifest, scene or JAR, and it makes no visible-truth claim.
    """
    module = _mask_runner()
    frame = _fake_frame()
    for name, pinned_domain in sorted(PINNED_TEMPLATE_2.items()):
        mask = module._load_mask(
            TRACKED_TEMPLATE_2 / f"{Path(name).stem}.json", frame.shape)
        assert module.mask_domain_sha256(mask) == pinned_domain, name


def test_a_json_mask_specification_reproduces_the_accepted_evaluator_domain(tmp_path):
    """The A2 loader uses the same rectangle/polygon semantics as the benchmark."""
    import sys as _sys

    scripts = str(REPO_ROOT / "scripts")
    if scripts not in _sys.path:
        _sys.path.insert(0, scripts)
    from benchmark_templates import _load_usable_masks, _mask_domain_sha256

    module = _mask_runner()
    spec = tmp_path / "frame.json"
    _write_spec(spec, rectangles=[(0, 0, 2, 2)],
                polygons=(((4, 4), (7, 4), (5, 7)),), size=(8, 8))
    frame = np.zeros((8, 8), dtype=np.uint8)
    observed = module._load_mask(spec, frame.shape)
    expected = _load_usable_masks({"frame.png": spec}, {"frame.png": frame})
    assert expected is not None
    assert np.array_equal(observed, expected["frame.png"])
    assert module.mask_domain_sha256(observed) == \
        _mask_domain_sha256(expected["frame.png"])


def test_a_png_mask_is_still_supported_when_no_specification_exists(tmp_path):
    from PIL import Image

    module = _mask_runner()
    array = np.zeros((6, 6), dtype=np.uint8)
    array[1:3, 2:5] = 255
    png = tmp_path / "frame.png"
    Image.fromarray(array, mode="L").save(png)
    spec = tmp_path / "frame.json"
    assert module.resolve_mask_path(tmp_path, "frame.png") == png
    assert module.resolve_mask_path(tmp_path, "absent.png") is None
    mask = module._load_mask(png, (6, 6))
    assert mask.dtype == bool
    assert mask.sum() == 6  # white keeps a pixel
    _write_spec(spec, rectangles=[(0, 0, 1, 1)], size=(6, 6))
    assert module.resolve_mask_path(tmp_path, "frame.png") == spec


def test_the_mask_loader_refuses_malformed_or_out_of_bounds_specifications(tmp_path):
    module = _mask_runner()
    frame_shape = (8, 8)
    out_of_bounds = tmp_path / "oob.json"
    _write_spec(out_of_bounds, rectangles=[(0, 0, 99, 99)], size=(8, 8))
    with pytest.raises(attribution.AttributionRefusal):
        module._load_mask(out_of_bounds, frame_shape)
    bad_polygon = tmp_path / "poly.json"
    bad_polygon.write_text(json.dumps({
        "schema": 2, "image_size": [8, 8],
        "excluded_rectangles": [], "excluded_polygons": [[[0, 0], [1, 1]]],
    }), encoding="utf-8")
    with pytest.raises(attribution.AttributionRefusal):
        module._load_mask(bad_polygon, frame_shape)
    wrong_size = tmp_path / "size.json"
    _write_spec(wrong_size, size=(4, 4))
    with pytest.raises(attribution.AttributionRefusal):
        module._load_mask(wrong_size, frame_shape)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{\"schema\": 2}", encoding="utf-8")
    with pytest.raises(attribution.AttributionRefusal):
        module._load_mask(malformed, frame_shape)
    assert module.resolve_mask_path(tmp_path, "missing.png") is None


def test_reviewed_pins_are_mandatory_and_a_domain_mismatch_fails_closed(
        monkeypatch, tmp_path):
    module = _mask_runner()
    frame = _fake_frame((8, 8))
    spec = tmp_path / "mask" / "frame.json"
    spec.parent.mkdir()
    _write_spec(spec, rectangles=[(0, 0, 2, 2)], size=(8, 8))
    frame_path = tmp_path / "frame.png"
    _write_frame_png(frame_path, (8, 8))
    reference = tmp_path / "ref.litematic"
    reference.write_bytes(b"a frozen reference stand-in")
    domain = module.mask_domain_sha256(module._load_mask(spec, frame.shape))
    pins = {
        "reference_sha256": module._sha256_bytes(reference.read_bytes()),
        "frames": [{"name": "frame.png",
                    "sha256": module._sha256_bytes(frame_path.read_bytes())}],
        "usable_masks": [{"frame": "frame.png",
                          "domain_sha256": domain}],
    }
    monkeypatch.setattr(module, "load_litematic",
                        lambda path: SimpleNamespace(blocks={(0, 0, 0): STONE}))
    monkeypatch.setattr(module, "load_scene_transform", lambda path: {})
    monkeypatch.setattr(module, "load_camera_manifest",
                        lambda path: {"frame.png": object()})
    monkeypatch.setattr(module, "visible_cells",
                        lambda blocks, scenes, cameras, resolutions,
                        usable_masks=None: {(0, 0, 0): 1})

    def load(require_pins=None, pins=pins, domain_sha256=None):
        pinned = {
            "reference_sha256": pins["reference_sha256"],
            "frames": pins["frames"],
            "usable_masks": [{"frame": "frame.png",
                              "domain_sha256": domain_sha256 or domain}],
        }
        return module.load_frozen_truth(
            fixture=tmp_path, reference=reference, scene_path=tmp_path / "scene.json",
            cameras_path=tmp_path / "cameras.json", mask_dir=spec.parent,
            filter_frames={"frame.png"}, pins=pinned,
            frame_paths={"frame.png": frame_path}, require_pins=require_pins)

    truth = load()
    assert truth["mask_origin"] == "tracked_json_specification"
    assert truth["mask_pinned_domains_checked"] == 1
    assert truth["mask_pins_match"] is True
    assert truth["caveats"] == []
    # The default is fail-closed whenever reviewed pins are supplied.
    with pytest.raises(attribution.AttributionRefusal):
        load(domain_sha256="0" * 64)
    # An explicit diagnostic downgrade still records the mismatch as a caveat.
    downgraded = load(require_pins=False, domain_sha256="0" * 64)
    assert downgraded["mask_pins_match"] is False
    assert downgraded["caveats"]
    # A pinned run that checks no pinned mask domain at all is not acceptable.
    with pytest.raises(attribution.AttributionRefusal):
        module.load_frozen_truth(
            fixture=tmp_path, reference=reference,
            scene_path=tmp_path / "scene.json",
            cameras_path=tmp_path / "cameras.json", mask_dir=None,
            filter_frames={"frame.png"}, pins=pins,
            frame_paths={"frame.png": frame_path})


def test_the_runner_fails_closed_end_to_end_when_pinned_domains_do_not_match(
        monkeypatch, capsys, tmp_path):
    module = _mask_runner()
    spec = tmp_path / "mask" / "frame.json"
    spec.parent.mkdir()
    _write_spec(spec, rectangles=[(0, 0, 2, 2)], size=(8, 8))
    frame = tmp_path / "frame.png"
    _write_frame_png(frame, (8, 8))
    reference = tmp_path / "ref.litematic"
    reference.write_bytes(b"a frozen reference stand-in")
    pin_manifest = tmp_path / "pins.json"
    pin_manifest.write_text(json.dumps({
        "manifest_schema": 1,
        "roles": {"role": {
            "fixture": "fixture", "reviewed": True,
            "reference_sha256": module._sha256_bytes(reference.read_bytes()),
            "frames": [{"name": "frame.png",
                        "sha256": module._sha256_bytes(frame.read_bytes())}],
            "usable_masks": [{"frame": "frame.png", "domain_sha256": "0" * 64}],
        }},
    }), encoding="utf-8")
    # Only the evaluator loaders that are outside this contract are stubbed; the
    # reference, frame and mask paths are read from disk by the real loader.
    monkeypatch.setattr(module, "load_litematic",
                        lambda path: SimpleNamespace(blocks={(0, 0, 0): STONE}))
    monkeypatch.setattr(module, "load_scene_transform", lambda path: {})
    monkeypatch.setattr(module, "load_camera_manifest",
                        lambda path: {"frame.png": object()})
    # Production now runs before any evaluator truth is opened, so the route must be
    # stubbed to "succeed" for the pin-mismatch fail-closed path to be reached.
    monkeypatch.setattr(module, "run_route_a", lambda **kwargs: (object(), None))
    report = tmp_path / "report.json"
    exit_code = module.main([
        "--route", "route-a", "--image", str(frame),
        "--reference", str(reference), "--scene", str(tmp_path / "scene.json"),
        "--cameras", str(tmp_path / "cameras.json"), "--masks", str(spec.parent),
        "--pins", str(pin_manifest), "--fixture-role", "role",
        "--report", str(report),
    ])
    assert exit_code == 2
    assert not report.exists()
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["attribution_available"] is False
    assert refusal["attribution"] is None
    # The refusal is the mask pin mismatch, not an unrelated missing input.
    assert "does not match the reviewed pin" in refusal["detail"]

